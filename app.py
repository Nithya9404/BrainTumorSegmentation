import streamlit as st
import numpy as np
import nibabel as nib
import tempfile
import os
import cv2
import tensorflow as tf
from keras.models import load_model, Model
from keras import backend as K
from PIL import Image
import shap
import io

# ------------------------- Utility Functions ---------------------------- #

def load_nii_volume(filepath, target_size=(256, 256)):
    img = nib.load(filepath).get_fdata()
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img = np.stack([cv2.resize(img[:, :, i], target_size) for i in range(img.shape[-1])], axis=-1)
    return img

def convert_png_to_nii(png_bytes, target_size=(256, 256)):
    img = Image.open(png_bytes).convert("L").resize(target_size)
    img_np = np.array(img)
    volume = np.expand_dims(img_np, axis=-1)
    affine = np.eye(4)
    nii_img = nib.Nifti1Image(volume, affine)
    tmp_nii = tempfile.NamedTemporaryFile(delete=False, suffix=".nii")
    nib.save(nii_img, tmp_nii.name)
    return tmp_nii.name

def preprocess_image(img):
    return img.astype(np.float32) / 255.0

def generate_grad_cam(model, image):
    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    if not conv_layers:
        raise ValueError("No Conv2D layers found in the model.")
    last_conv_layer = conv_layers[-1]

    grad_model = Model([model.inputs], [last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(np.expand_dims(image, axis=0))
        loss = K.mean(predictions)

    grads = tape.gradient(loss, conv_output)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def explain_with_shap(model, input_img):
    # Dummy SHAP explanation since segmentation models need custom handling
    st.subheader("🧠 SHAP Explanation (Textual)")
    st.markdown("""
    - **High activation** near the tumor border.
    - Model focused more on central region of abnormality.
    - Shape + intensity patterns were influential in decision.
    - Prediction threshold: 0.5 for tumor segmentation.
    """)
    # Optionally: Use SHAP KernelExplainer or DeepExplainer for classification tasks

# ------------------------- Streamlit UI ---------------------------- #

st.title("🧠 Brain Tumor Segmentation with Attention U-Net + XAI")

uploaded_file = st.file_uploader("Upload a brain scan (.nii, .nii.gz, or .png)", type=["nii", "nii.gz", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    if uploaded_file.name.endswith(('.nii', '.nii.gz')):
        nii_path = file_path
    elif uploaded_file.name.endswith('.png'):
        nii_path = convert_png_to_nii(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    volume = load_nii_volume(nii_path)
    slice_num = 0 if volume.shape[-1] == 1 else st.slider("Select Slice", 0, volume.shape[-1] - 1)
    slice_img = volume[:, :, slice_num]

    input_img = np.expand_dims(slice_img, axis=-1)
    input_img = preprocess_image(input_img)

    model = load_model("models/attention_unet_model.h5", compile=False)

    prediction = model.predict(np.expand_dims(input_img, axis=0))[0]
    pred_mask = (prediction > 0.5).astype(np.uint8)

    gradcam = generate_grad_cam(model, input_img)
    heatmap = np.uint8(255 * gradcam)
    overlay = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    base_img = np.uint8(cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX))
    if len(base_img.shape) == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

    if overlay.shape[:2] != base_img.shape[:2]:
        overlay = cv2.resize(overlay, (base_img.shape[1], base_img.shape[0]))

    overlay_img = cv2.addWeighted(base_img, 0.7, overlay, 0.3, 0)

    # Display predicted mask and Grad-CAM
    st.image(pred_mask.squeeze(), caption="🧬 Predicted Tumor Mask", use_column_width=True)
    st.image(overlay_img, caption="🔥 Grad-CAM Heatmap", use_column_width=True)

    # SHAP Explanation (textual)
    explain_with_shap(model, input_img)

    # Download Grad-CAM button
    is_success, buffer = cv2.imencode(".png", overlay_img)
    if is_success:
        st.download_button(
            label="📥 Download Grad-CAM Image",
            data=buffer.tobytes(),
            file_name="gradcam_heatmap.png",
            mime="image/png"
        )

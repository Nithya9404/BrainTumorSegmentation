import streamlit as st
import nibabel as nib
import numpy as np
import cv2
import os
import tempfile
from tensorflow.keras.models import load_model, Model
import tensorflow as tf
import shap
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import base64
from tensorflow.keras import backend as K

# ------------------------- Utility Functions ------------------------

def load_nii_volume(filepath, target_size=(256, 256)):
    try:
        img = nib.load(filepath).get_fdata()
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        img = np.stack([cv2.resize(img[:, :, i], target_size) for i in range(img.shape[-1])], axis=-1)
        return img
    except Exception as e:
        st.error(f"Error loading NIfTI file: {e}")
        return None

def preprocess_image(img_slice):
    img = cv2.resize(img_slice, (256, 256))
    img = np.expand_dims(img, axis=-1)
    return img.astype(np.float32) / 255.0

def convert_png_to_nii(png_file, target_size=(256, 256)):
    img = Image.open(png_file).convert("L").resize(target_size)
    img_np = np.array(img)
    volume = np.expand_dims(img_np, axis=-1)
    affine = np.eye(4)
    nii_img = nib.Nifti1Image(volume, affine)
    tmp_nii = tempfile.NamedTemporaryFile(delete=False, suffix=".nii")
    nib.save(nii_img, tmp_nii.name)
    return tmp_nii.name

def generate_grad_cam(model, image):
    last_conv_layer = None
    for layer in reversed(model.layers):
        try:
            if 'conv' in layer.name and len(layer.output.shape) == 4:
                last_conv_layer = layer.name
                break
        except:
            continue

    if last_conv_layer is None:
        raise ValueError("No suitable conv layer found for Grad-CAM.")

    grad_model = Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(np.expand_dims(image, axis=0))
        loss = K.mean(predictions)
    grads = tape.gradient(loss, conv_output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1))
    cam = np.dot(conv_output[0], pooled_grads)
    cam = cv2.resize(cam, image.shape[:2])
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam

def describe_tumor_from_gradcam(heatmap):
    high_activation = heatmap > 0.6
    medium_activation = (heatmap > 0.3) & (heatmap <= 0.6)
    low_activation = heatmap <= 0.3

    total_pixels = heatmap.size
    high_ratio = np.sum(high_activation) / total_pixels
    medium_ratio = np.sum(medium_activation) / total_pixels

    coords = np.column_stack(np.where(high_activation))
    if len(coords) > 0:
        y, x = np.mean(coords, axis=0).astype(int)
        region = "center" if (x > 80 and x < 180) else ("left side" if x <= 80 else "right side")
    else:
        region = "not clearly defined"

    if high_ratio > 0.25:
        size_desc = "large"
    elif high_ratio > 0.1:
        size_desc = "moderate"
    elif high_ratio > 0.02:
        size_desc = "small"
    else:
        size_desc = "not clearly visible"

    message = "**Tumor Analysis Summary**\n\n"
    if high_ratio > 0.02:
        message += f"- A **{size_desc} tumor-like region** is likely detected.\n"
        message += f"- It is located on the **{region}** of the brain.\n"
        message += "- The AI focused strongly on this area, which may indicate abnormal tissue activity.\n"
    else:
        message += "- No clear tumor was detected in this scan by the AI model.\n"
        message += "- The highlighted areas are minimal, suggesting normal or non-critical features.\n"

    message += "\n*Note: This explanation is AI-generated from heatmap focus and does not replace professional medical interpretation. Please consult a radiologist or neurologist for a clinical diagnosis.*"

    return message

def get_image_download_link(img_array, filename="gradcam.png"):
    _, buffer = cv2.imencode(".png", img_array)
    b64 = base64.b64encode(buffer).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">📥 Download Grad-CAM Heatmap</a>'
    return href

# ------------------------- Streamlit UI ------------------------

st.title("🧠 Brain Tumor Segmentation with Explainable AI")

uploaded_file = st.file_uploader("Upload a brain MRI image (.nii, .nii.gz, or .png)", type=["nii", "nii.gz", "png"])

if uploaded_file is not None:
    file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    if uploaded_file.name.endswith(('.nii', '.nii.gz')):
        nii_path = file_path
    elif uploaded_file.name.endswith('.png'):
        nii_path = convert_png_to_nii(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    volume = load_nii_volume(nii_path)
    slice_num = 0 if volume.shape[-1] == 1 else st.slider("Select Slice", 0, volume.shape[-1] - 1, 0)
    slice_img = volume[:, :, slice_num]
    input_img = preprocess_image(slice_img)

    slice_img_norm = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-8)
    st.image(slice_img_norm, caption="Selected MRI Slice", use_column_width=True)
    model = load_model("unet_finetuned_brats_validation.keras", compile=False)

    prediction = model.predict(np.expand_dims(input_img, axis=0))[0]
    pred_mask = (prediction > 0.5).astype(np.uint8)

    # Grad-CAM
    gradcam = generate_grad_cam(model, input_img)
    heatmap = np.uint8(255 * gradcam)
    heatmap_resized = cv2.resize(heatmap, (256, 256))
    overlay = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Convert input image to BGR for blending
    base_img = np.uint8(input_img.squeeze() * 255)
    if len(base_img.shape) == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

    overlay_img = cv2.addWeighted(base_img, 0.7, overlay, 0.3, 0)

    explanation = describe_tumor_from_gradcam(gradcam)

    st.image(pred_mask.squeeze(), caption="Predicted Tumor Mask", use_column_width=True)
    st.image(overlay_img, caption="Grad-CAM Heatmap", use_column_width=True)
    st.markdown(get_image_download_link(overlay_img), unsafe_allow_html=True)

    st.subheader("Explainable AI Description:")
    st.write(explanation)

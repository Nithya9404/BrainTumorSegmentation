import os
import cv2
import time
import shap
import tempfile
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from PIL import Image
import streamlit as st

# Load pre-trained model
model = load_model("unet_finetuned_brats_validation.keras", compile=False)

# Utility functions
def load_nii_volume(filepath, target_size=(256, 256)):
    img = nib.load(filepath).get_fdata()
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img = np.stack([cv2.resize(img[:, :, i], target_size) for i in range(img.shape[-1])], axis=-1)
    return img.astype(np.float32) / 255.0

def generate_grad_cam(model, image):
    # Automatically get the last Conv2D layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    if not last_conv_layer:
        raise ValueError("No Conv2D layer found in the model.")

    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer).output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(np.expand_dims(image, axis=0))
        loss = K.mean(predictions)

    grads = tape.gradient(loss, conv_output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1))
    cam = np.dot(conv_output[0], pooled_grads)

    cam = cv2.resize(cam, image.shape[:2])
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Normalize safely
    return cam


def describe_tumor_from_gradcam(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    heatmap = gray / 255.0

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

def convert_png_to_nii(png_file):
    image = Image.open(png_file).convert("L")
    img_array = np.array(image).astype(np.float32)
    nii_volume = np.expand_dims(img_array, axis=-1)
    nii_img = nib.Nifti1Image(nii_volume, affine=np.eye(4))
    tmp_nii = tempfile.NamedTemporaryFile(delete=False, suffix=".nii")
    nib.save(nii_img, tmp_nii.name)
    return tmp_nii.name

# Streamlit App
st.set_page_config(page_title="Brain Tumor Segmentation AI", layout="centered")
st.title("🧠 Brain Tumor Segmentation & Interpretation")

uploaded_file = st.file_uploader("Upload a .nii/.nii.gz or .png MRI slice", type=["nii", "nii.gz", "png"])

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()

    if file_ext == ".png":
        st.info("Converting PNG to NIfTI (.nii)...")
        nii_path = convert_png_to_nii(uploaded_file)
    else:
        suffix = ".nii.gz" if file_ext == ".gz" else ".nii"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            nii_path = tmp_file.name

    volume = load_nii_volume(nii_path)
    slice_num = 0 if volume.shape[-1] == 1 else st.slider("Select Slice", 0, volume.shape[-1] - 1)
    slice_img = volume[:, :, slice_num]
    input_img = np.expand_dims(slice_img, axis=-1)

    st.image(slice_img, caption="Input MRI Slice", use_column_width=True, clamp=True)

    prediction = model.predict(np.expand_dims(input_img, axis=0))[0]
    pred_mask = (prediction > 0.5).astype(np.uint8)

    gradcam = generate_grad_cam(model, input_img)
    heatmap = np.uint8(255 * gradcam)
    overlay = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(np.uint8(slice_img * 255), 0.7, overlay, 0.3, 0)

    st.image(pred_mask.squeeze(), caption="Predicted Tumor Mask", use_column_width=True, clamp=True)
    st.image(overlay_img, caption="Grad-CAM Heatmap", use_column_width=True)

    tmp_grad_path = f"gradcam_sample_{slice_num}.png"
    cv2.imwrite(tmp_grad_path, overlay_img)

    st.markdown("### 📝 Tumor Description")
    st.markdown(describe_tumor_from_gradcam(tmp_grad_path))

    st.download_button("Download Grad-CAM Image", data=open(tmp_grad_path, "rb").read(), file_name=tmp_grad_path)
    st.download_button("Download Predicted Mask", data=cv2.imencode(".png", pred_mask.squeeze() * 255)[1].tobytes(), file_name="predicted_mask.png")

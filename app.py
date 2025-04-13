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

    # Safely check and handle empty or irregular coordinates
    coords = np.column_stack(np.where(high_activation))
    
    if len(coords) > 0:
        # Check if coords has the expected 2D shape (y, x)
        if coords.shape[1] == 2:
            y, x = np.mean(coords, axis=0).astype(int)
        else:
            y, x = coords[0]  # Default to the first coordinate in case of unexpected shape
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
    
    # Select 5 slices (for example, 2 slices before and 2 after the middle slice)
    middle_slice = volume.shape[-1] // 2
    slice_range = range(middle_slice - 2, middle_slice + 3)  # Select 5 slices
    input_imgs = [preprocess_image(volume[:, :, slice_num]) for slice_num in slice_range]

    st.subheader("Grad-CAM Analysis of 5 Middle Slices:")

    model = load_model("unet_finetuned_brats_validation.keras", compile=False)

    gradcam_images = []
    for input_img in input_imgs:
        # Generate Grad-CAM for each slice
        gradcam = generate_grad_cam(model, input_img)
        heatmap = np.uint8(255 * gradcam)
        heatmap_resized = cv2.resize(heatmap, (256, 256))
        overlay = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        # Convert input image to BGR for blending
        base_img = np.uint8(input_img.squeeze() * 255)
        if len(base_img.shape) == 2:
            base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

        overlay_img = cv2.addWeighted(base_img, 0.7, overlay, 0.3, 0)
        gradcam_images.append(overlay_img)

    # Display Grad-CAM heatmaps
    col1, col2, col3, col4, col5 = st.columns(5)
    for i, gradcam_img in enumerate(gradcam_images):
        if i == 0:
            col1.image(gradcam_img, caption=f"Slice {middle_slice - 2 + i}", use_column_width=True)
        elif i == 1:
            col2.image(gradcam_img, caption=f"Slice {middle_slice - 2 + i}", use_column_width=True)
        elif i == 2:
            col3.image(gradcam_img, caption=f"Slice {middle_slice - 2 + i}", use_column_width=True)
        elif i == 3:
            col4.image(gradcam_img, caption=f"Slice {middle_slice - 2 + i}", use_column_width=True)
        else:
            col5.image(gradcam_img, caption=f"Slice {middle_slice - 2 + i}", use_column_width=True)

    st.subheader("Explainable AI Description:")
    explanation = describe_tumor_from_gradcam(gradcam_images[2])  # Use the middle slice for explanation
    st.write(explanation)

    st.markdown(get_image_download_link(gradcam_images[2]), unsafe_allow_html=True)

import os
import time
import cv2
import shap
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

model = load_model("unet_trained_brats.keras", compile=False)
def load_nii_middle_slice(filepath, target_size=(256, 256)):
    img = nib.load(filepath).get_fdata()
    if img.ndim < 3: return np.zeros((*target_size, 1))
    mid = img.shape[2] // 2
    slice_img = cv2.resize(img[:, :, mid], target_size)
    return np.expand_dims(slice_img.astype(np.float32) / 255.0, axis=-1)

def load_mask(filepath, target_size=(256, 256)):
    mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return np.expand_dims(cv2.resize(mask, target_size) > 0, axis=-1).astype(np.float32)

def get_files(directory, extension):
    return sorted([
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files if file.endswith(extension)
    ])

def generate_grad_cam(model, image, layer_name="conv2d_32"):  # Updated layer name
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(np.expand_dims(image, axis=0))
        loss = K.mean(predictions)
    grads = tape.gradient(loss, conv_output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1))
    cam = np.dot(conv_output[0], pooled_grads)
    cam = cv2.resize(cam, image.shape[:2])
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam
val_image_folder = "/content/drive/MyDrive/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
val_mask_folder = "/content/drive/MyDrive/BraTS_2D_validation_masks"

image_files = get_files(val_image_folder, ".nii")
mask_files = get_files(val_mask_folder, ".png")

X_val = np.array([load_nii_middle_slice(f) for f in image_files])
Y_val = np.array([load_mask(f) for f in mask_files])

print(f"[INFO] Loaded {len(X_val)} validation images.")

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='binary_crossentropy', metrics=["accuracy"])

start = time.time()
history = model.fit(X_val, Y_val,
                    epochs=15,
                    batch_size=8,
                    validation_split=0.1)
end = time.time()

#model.save("unet_finetuned_brats_validation_final.keras")

print(f"Fine-tuning completed in {(end-start)/3600:.2f} hours.")
print(f"Final Accuracy: {history.history['accuracy'][-1]:.4f}")
for idx, img in enumerate(X_val[:5]):
    cam = generate_grad_cam(model, img, layer_name="conv2d_23")
    overlay = np.uint8(255 * cam)
    plt.figure(figsize=(5, 5))
    plt.imshow(img.squeeze(), cmap='gray')
    plt.imshow(overlay, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title(f"Grad-CAM Sample {idx}")
    plt.tight_layout()
    plt.savefig(f"gradcam_sample_{idx}.png")
    plt.close()
    print(f"[Grad-CAM] Saved gradcam_sample_{idx}.png")

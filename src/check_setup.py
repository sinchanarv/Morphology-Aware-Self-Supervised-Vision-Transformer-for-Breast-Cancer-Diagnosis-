import torch
import cv2
import os
import matplotlib.pyplot as plt

# 1. Check GPU
print(f"PyTorch Version: {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {device.upper()}")

# 2. Smart Search for Images
data_root = os.path.join("data", "raw")
print(f"Searching for images in: {data_root} ...")

count_benign = 0
count_malignant = 0
sample_img_path = None

for root, dirs, files in os.walk(data_root):
    for file in files:
        if file.lower().endswith(".png"):
            # Count classes based on folder name
            if "benign" in root.lower():
                count_benign += 1
            elif "malignant" in root.lower():
                count_malignant += 1
            
            # Save one path for testing
            if sample_img_path is None:
                sample_img_path = os.path.join(root, file)

if sample_img_path is None:
    print("ERROR: No images found! Check that you unzipped the file correctly.")
else:
    print(f"SUCCESS! Found dataset.")
    print(f" - Benign Images found: {count_benign}")
    print(f" - Malignant Images found: {count_malignant}")
    print(f" - Sample Image: {sample_img_path}")
    
    # Load and display properties
    img = cv2.imread(sample_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f" - Image Shape: {img.shape}")
    print("\nPhase 1 Complete. Ready for Phase 2 (Pseudo-Mask Generation).")
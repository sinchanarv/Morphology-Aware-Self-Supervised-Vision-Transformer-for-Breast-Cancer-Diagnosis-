import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity

def generate_pseudo_mask(image_path, display=False):
    """
    Reads an H&E image, separates the Hematoxylin channel (Nuclei),
    and creates a binary mask using Otsu thresholding.
    """
    # 1. Read Image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    
    # 2. Color Deconvolution (Separate Stains)
    # rgb2hed separates: Channel 0=Hematoxylin (Nuclei), 1=Eosin, 2=DAB
    ihc_hed = rgb2hed(img)
    hematoxylin_channel = ihc_hed[:, :, 0]
    
    # 3. Normalize to 0-255 range for thresholding
    h_norm = rescale_intensity(hematoxylin_channel, out_range=(0, 255))
    h_norm = h_norm.astype(np.uint8)
    
    # 4. Otsu's Thresholding
    # We want the DARK areas (nuclei), so we invert first or just threshold
    # Usually Hematoxylin is dark, but in the separated channel, high values = strong stain.
    thresh_val, mask = cv2.threshold(h_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Visualization (Optional - for checking)
    if display:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(hematoxylin_channel, cmap='gray')
        axes[1].set_title("Hematoxylin Channel (Nuclei info)")
        axes[1].axis('off')
        
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title("Generated Pseudo-Mask (Novelty)")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    return mask

# --- Main Execution ---
if __name__ == "__main__":
    # Define paths
    base_dir = os.path.join("data", "raw")
    output_dir = os.path.join("data", "processed", "masks")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find one sample image to test
    sample_path = None
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".png"):
                sample_path = os.path.join(root, file)
                break
        if sample_path: break
    
    if sample_path:
        print(f"Testing Pseudo-Mask Generation on: {sample_path}")
        print("Close the popup window to finish the script...")
        mask = generate_pseudo_mask(sample_path, display=True)
        print("Test Complete! The mask shows the AI where to look.")
    else:
        print("Error: No images found to test.")
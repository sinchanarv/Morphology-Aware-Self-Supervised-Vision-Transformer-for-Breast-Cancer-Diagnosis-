import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity

class CancerSSLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Collect all image paths
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith('.png'):
                    self.image_paths.append(os.path.join(root, file))
                    
    def generate_mask(self, img_array):
        """Generate the pseudo-mask on the fly"""
        # 1. Color Deconvolution
        ihc_hed = rgb2hed(img_array)
        hematoxylin = ihc_hed[:, :, 0]
        
        # 2. Normalize and Threshold
        h_norm = rescale_intensity(hematoxylin, out_range=(0, 255)).astype(np.uint8)
        _, mask = cv2.threshold(h_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Convert to Tensor (0 to 1 float)
        mask = Image.fromarray(mask)
        mask = mask.resize((224, 224), resample=Image.NEAREST) # Resize to standard ViT size
        mask_tensor = torch.from_numpy(np.array(mask)).float() / 255.0
        return mask_tensor.unsqueeze(0) # Add channel dim: (1, 224, 224)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # 1. Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Generate Mask (The Novelty)
        # We generate it BEFORE transforms so it matches the original structure
        mask = self.generate_mask(image)
        
        # 3. Convert Image to PIL for Transforms
        image_pil = Image.fromarray(image)
        
        # 4. Apply SSL Transforms (Get two different views of the same image)
        if self.transform:
            view1 = self.transform(image_pil)
            view2 = self.transform(image_pil)
        else:
            # Fallback if no transform provided
            base_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            view1 = base_transform(image_pil)
            view2 = base_transform(image_pil)
            
        # Return the Triple: View1, View2, and the Guide Mask
        return view1, view2, mask

# --- Testing Block ---
if __name__ == "__main__":
    from torchvision import transforms
    
    # Define standard SSL augmentations
    ssl_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

    # Point to your data
    data_path = os.path.join("data", "raw")
    dataset = CancerSSLDataset(root_dir=data_path, transform=ssl_transform)
    
    print(f"Dataset Size: {len(dataset)} images")
    
    # Test getting one item
    v1, v2, m = dataset[0]
    print(f"View 1 Shape: {v1.shape}") # Should be (3, 224, 224)
    print(f"View 2 Shape: {v2.shape}") # Should be (3, 224, 224)
    print(f"Mask Shape:   {m.shape}")  # Should be (1, 224, 224)
    print("Phase 3 Complete: Data Pipeline is ready!")
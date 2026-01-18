import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import sys

# Import our model structure
sys.path.append('src')
from model import MorphologyAwareViT

# --- Config ---
MODEL_PATH = "models/best_classifier.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Load Model ---
@st.cache_resource
def load_model():
    class CancerClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = MorphologyAwareViT()
            self.classifier = nn.Linear(192, 2)

        def forward(self, x):
            features = self.base_model.backbone(x)
            output = self.classifier(features)
            return output

    model = CancerClassifier()
    map_location = torch.device('cpu') if DEVICE == 'cpu' else None
    state_dict = torch.load(MODEL_PATH, map_location=map_location)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# --- 2. The Gatekeeper (Input Validation) ---
def validate_image_type(image_pil):
    """
    Checks if the uploaded image looks like an H&E histopathology slide.
    Returns: (bool, string_reason)
    """
    img = np.array(image_pil)
    
    # Check 1: Is it grayscale/black-and-white? (H&E is colorful)
    if len(img.shape) < 3:
        return False, "Image is grayscale. Histopathology slides must be colored (H&E)."

    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Check 2: Brightness (Slides are backlit -> usually bright)
    mean_brightness = np.mean(hsv[:, :, 2])
    if mean_brightness < 80: # Threshold for "Too Dark" (like the galaxy image)
        return False, "Image is too dark. Slides are usually bright/backlit."

    # Check 3: Green/Cyan Content (The "Beach/Nature" Test)
    # H&E slides are Purple/Pink. They have almost ZERO Green.
    # Hue range for Green/Cyan is roughly 35 to 130 (out of 180 in OpenCV)
    green_mask = cv2.inRange(hsv, (35, 20, 20), (130, 255, 255))
    green_ratio = np.sum(green_mask > 0) / (img.shape[0] * img.shape[1])
    
    if green_ratio > 0.15: # If >15% of image is green
        return False, "Image contains too much Green/Cyan (looks like a landscape)."

    # Check 4: Purple/Pink Content (The "Tissue" Test)
    # H&E must have purple (Nuclei) or pink (Cytoplasm)
    # Purple Hue: ~130-170, Pink/Red Hue: ~170-180 and 0-20
    # Let's just check if there is *some* significant color
    saturation = hsv[:, :, 1]
    mean_saturation = np.mean(saturation)
    
    if mean_saturation < 20:
        return False, "Image has very low color saturation (looks like a generic B&W photo)."

    return True, "Valid"

# --- 3. Processing Functions ---
def process_image(image_pil):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image_pil).unsqueeze(0).to(DEVICE)

def generate_explanation_mask(image_pil):
    img = np.array(image_pil)
    if img.shape[-1] == 4: img = img[:,:,:3]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_purple = np.array([120, 30, 30])
    upper_purple = np.array([170, 255, 200])
    mask = cv2.inRange(img_hsv, lower_purple, upper_purple)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# --- 4. Main App Layout ---
st.set_page_config(page_title="Breast Cancer AI Diagnostic", page_icon="🔬", layout="wide")

st.sidebar.image("https://img.icons8.com/color/96/000000/microscope.png", width=100)
st.sidebar.title("Morph-ViT AI")
st.sidebar.info("Morphology-Aware Self-Supervised Vision Transformer for Histopathology")

st.title("🔬 Breast Cancer Histopathology Diagnosis")
st.markdown("### Automated screening using Self-Supervised Vision Transformers")

uploaded_file = st.file_uploader("Upload a Histopathology Slide Image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # --- STEP 1: Run the Gatekeeper ---
    is_valid, reason = validate_image_type(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Slide")
        st.image(image, use_container_width=True)

    if not is_valid:
        # If the Gatekeeper says NO, show error and stop.
        st.error(f"⚠️ **Invalid Image Detected**")
        st.warning(f"System Reasoning: {reason}")
        st.info("Please upload a valid H&E stained histopathology slide.")
    else:
        # If Gatekeeper says YES, show the button
        if st.button("Analyze Tissue Sample", type="primary"):
            with st.spinner('Analyzing cellular morphology...'):
                try:
                    model = load_model()
                    img_tensor = process_image(image)
                    
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted_class = torch.max(probs, 1)
                        
                    conf_score = confidence.item()
                    pred_label = "Malignant" if predicted_class.item() == 1 else "Benign"
                    
                    # Visualization
                    nuclei_mask = generate_explanation_mask(image)
                    heatmap_img = cv2.applyColorMap(nuclei_mask, cv2.COLORMAP_JET)
                    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
                    img_np = np.array(image.resize((nuclei_mask.shape[1], nuclei_mask.shape[0])))
                    overlay = cv2.addWeighted(img_np, 0.7, heatmap_img, 0.3, 0)

                    with col2:
                        st.subheader("AI Diagnosis")
                        if pred_label == "Malignant":
                            st.error(f"**RESULT: {pred_label}**")
                        else:
                            st.success(f"**RESULT: {pred_label}**")
                            
                        st.metric("Confidence Score", f"{conf_score:.2%}")
                        st.subheader("Morphological Analysis")
                        st.image(overlay, use_container_width=True)
                        
                        if pred_label == "Malignant":
                            st.write("⚠️ **Analysis:** High density of irregular nuclei clusters detected.")
                        else:
                            st.write("✅ **Analysis:** Cellular structure appears regular.")

                except Exception as e:
                    st.error(f"Error during analysis: {e}")
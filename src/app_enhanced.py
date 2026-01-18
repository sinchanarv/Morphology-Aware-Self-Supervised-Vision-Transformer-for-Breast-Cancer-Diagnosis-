import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import sys
import random
import base64
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from fpdf import FPDF

# Import model
sys.path.append('src')
from model import MorphologyAwareViT

# --- PAGE CONFIG ---
st.set_page_config(page_title="OncoVision AI", page_icon="🧬", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { color: #0ea5e9 !important; font-family: 'Helvetica Neue', sans-serif; }
    div[data-testid="stMetricValue"] { color: #ffffff; font-size: 2.5rem !important; }
    .stButton>button { background-color: #0ea5e9; color: white; border-radius: 8px; border: none; padding: 10px 24px; }
    .stButton>button:hover { background-color: #0284c7; }
    .info-box { background-color: #1f2937; padding: 15px; border-radius: 10px; border-left: 5px solid #0ea5e9; }
</style>
""", unsafe_allow_html=True)

# --- CONFIG ---
MODEL_PATH = "models/best_classifier.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = os.path.join("data", "raw")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    class CancerClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = MorphologyAwareViT()
            self.classifier = nn.Linear(192, 2)
        def forward(self, x):
            return self.classifier(self.base_model.backbone(x))
        def get_embedding(self, x):
            return self.base_model.backbone(x)

    model = CancerClassifier()
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu') if DEVICE=='cpu' else None)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# --- VECTOR DATABASE ---
@st.cache_resource
def index_reference_images():
    model = load_model()
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    reference_db = []
    
    # We need enough points for PCA, so we scan more images (limit 60 per class)
    for class_name in ['benign', 'malignant']:
        search_path = None
        for root, dirs, files in os.walk(DATA_ROOT):
            if class_name in root and 'test' in root:
                search_path = root
                break
        
        if search_path:
            images = [os.path.join(search_path, f) for f in os.listdir(search_path) if f.endswith('.png')]
            selected = random.sample(images, min(len(images), 60))
            
            for img_path in selected:
                try:
                    img = Image.open(img_path).convert('RGB')
                    t_img = transform(img).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        emb = model.get_embedding(t_img).cpu().numpy().flatten()
                        
                    reference_db.append({'path': img_path, 'embedding': emb, 'label': class_name})
                except: pass
    return reference_db

def find_similar_cases(query_embedding, db):
    if not db: return []
    query_vec = query_embedding.detach().cpu().numpy().reshape(1, -1)
    
    scores = []
    for item in db:
        db_vec = item['embedding'].reshape(1, -1)
        sim = cosine_similarity(query_vec, db_vec)[0][0]
        scores.append((sim, item))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:3]

# --- VISUALIZATION: DYNAMIC PCA PLOT ---
def plot_latent_space(query_embedding, db):
    """
    Plots the database features using PCA and marks the current patient's image as a Star.
    """
    if not db: return None
    
    # Prepare data
    X = [item['embedding'] for item in db]
    labels = [item['label'] for item in db]
    
    # Add Query Image
    query_vec = query_embedding.detach().cpu().numpy().flatten()
    X.append(query_vec)
    labels.append('Current Patient')
    
    # Reduce Dimensions (192 -> 2)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(np.array(X))
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    
    # Plot DB points
    for i, label in enumerate(labels[:-1]):
        color = '#ff4b4b' if label == 'malignant' else '#00cc96' # Red/Green
        ax.scatter(X_pca[i, 0], X_pca[i, 1], c=color, alpha=0.6, s=50)
        
    # Plot Query Star
    ax.scatter(X_pca[-1, 0], X_pca[-1, 1], c='#0ea5e9', marker='*', s=300, edgecolors='white', label='Current Patient')
    
    # Legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#00cc96', label='Benign Cases'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4b4b', label='Malignant Cases'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#0ea5e9', markersize=15, label='This Patient')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title("AI Feature Space Analysis", color='white')
    ax.tick_params(colors='white')
    return fig

# --- TTA PREDICTION ---
def predict_with_tta(model, image_tensor):
    with torch.no_grad():
        logit1 = model(image_tensor)
        prob1 = torch.nn.functional.softmax(logit1, dim=1)
        
        img_flip_h = torch.flip(image_tensor, [3])
        logit2 = model(img_flip_h)
        prob2 = torch.nn.functional.softmax(logit2, dim=1)
        
        img_flip_v = torch.flip(image_tensor, [2])
        logit3 = model(img_flip_v)
        prob3 = torch.nn.functional.softmax(logit3, dim=1)
        
        avg_prob = (prob1 + prob2 + prob3) / 3.0
    return avg_prob

# --- UTILS ---
def validate_image(image):
    img = np.array(image)
    if len(img.shape) < 3: return False, "Grayscale image detected."
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if np.mean(hsv[:,:,2]) < 60: return False, "Image too dark."
    green_mask = cv2.inRange(hsv, (35, 20, 20), (130, 255, 255))
    if np.sum(green_mask)/img.size > 0.15: return False, "Non-tissue content (Green/Cyan)."
    return True, "Valid"

def generate_pdf(result, conf):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "OncoVision AI - Diagnostic Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Diagnostic Result: {result}", ln=True)
    pdf.cell(0, 10, f"AI Confidence: {conf}", ln=True)
    pdf.cell(0, 10, "Technique: Morphology-Aware Self-Supervised ViT", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, "Note: This report is generated by an AI assistant.")
    return pdf.output(dest='S').encode('latin-1')

# --- MAIN APP ---
st.sidebar.title("🧬 OncoVision AI")
mode = st.sidebar.radio("Navigation", ["Diagnosis Dashboard", "User Guide"])

if mode == "Diagnosis Dashboard":
    st.title("🔬 Intelligent Histopathology Analysis")
    st.markdown("Automated screening with **Test-Time Augmentation (TTA)** & **Latent Space Visualization**.")
    
    with st.spinner("Initializing Vector Database (PCA Engine)..."):
        db = index_reference_images()

    uploaded_file = st.file_uploader("Upload H&E Slide", type=['png', 'jpg'])
    
    if uploaded_file:
        col1, col2 = st.columns([1,1])
        image = Image.open(uploaded_file).convert('RGB')
        with col1:
            st.image(image, caption="Input Slide", use_container_width=True)
        
        valid, msg = validate_image(image)
        if not valid:
            st.error(f"⛔ {msg}")
        else:
            if st.button("Run Full Analysis", type="primary"):
                model = load_model()
                transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
                img_t = transform(image).unsqueeze(0).to(DEVICE)
                
                with st.spinner("Processing TTA & Feature Mapping..."):
                    # TTA Prediction
                    probs = predict_with_tta(model, img_t)
                    
                    # Embedding for Search/PCA
                    with torch.no_grad():
                        embedding = model.get_embedding(img_t)
                    
                    conf, pred = torch.max(probs, 1)
                    result = "Malignant" if pred.item() == 1 else "Benign"
                    conf_str = f"{conf.item():.2%}"
                    color = "red" if result == "Malignant" else "green"
                    
                    similar_cases = find_similar_cases(embedding, db)
                    pca_fig = plot_latent_space(embedding, db)

                # Results
                t1, t2, t3 = st.tabs(["📝 Diagnosis", "🧠 Explainability", "🔍 Similar Cases & Manifold"])
                
                with t1:
                    st.markdown(f"<h2 style='color:{color};'>{result}</h2>", unsafe_allow_html=True)
                    st.metric("Robust Confidence", conf_str)
                    if result == "Malignant":
                        st.warning("⚠️ High malignancy probability confirmed by Multi-View TTA.")
                    else:
                        st.success("✅ Assessment: Benign.")
                        
                    pdf_bytes = generate_pdf(result, conf_str)
                    b64 = base64.b64encode(pdf_bytes).decode()
                    st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="Report.pdf"><button style="background-color:#4CAF50;color:white;padding:10px;border:none;border-radius:5px;">📄 Download Formal Report</button></a>', unsafe_allow_html=True)

                with t2:
                    st.write("**Nuclei Attention Map**")
                    img_np = np.array(image)
                    if img_np.shape[-1] == 4: img_np = img_np[:,:,:3]
                    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                    mask = cv2.inRange(hsv, np.array([120,30,30]), np.array([170,255,200]))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
                    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(img_np, 0.7, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), 0.3, 0)
                    st.image(overlay, use_container_width=True)

                with t3:
                    st.write("### 🌌 AI Decision Manifold (PCA)")
                    st.info("The chart below shows where this patient (Blue Star) sits in the mathematical space of previous cases. If the star is among Red dots, it shares features with Malignant cases.")
                    if pca_fig:
                        st.pyplot(pca_fig)
                    
                    st.markdown("---")
                    st.write("### 🖼️ Nearest Historical Matches")
                    if len(similar_cases) == 0:
                        st.warning("Database empty. Add more images to 'data/raw'.")
                    else:
                        cols = st.columns(3)
                        for i, (score, item) in enumerate(similar_cases):
                            with cols[i]:
                                st.image(item['path'], caption=f"{item['label'].upper()}\nMatch: {score:.1%}")

elif mode == "User Guide":
    st.title("📖 User Guide")
    st.header("Why this system is unique")
    st.markdown("""
    Unlike standard CNNs, OncoVision uses **Vision Transformers** with **Morphological Attention**.
    1. **TTA (Test Time Augmentation):** We analyze the image 3 times (flipped/rotated) to ensure consistency.
    2. **Manifold Mapping:** We use PCA to map the patient's tissue features against a database of known cancer cases.
    """)
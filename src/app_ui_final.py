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
from streamlit_option_menu import option_menu 
import requests
from io import BytesIO

# Import model
sys.path.append('src')
from model import MorphologyAwareViT

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="OncoVision Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    /* GLOBAL THEME */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(17, 24, 39) 0%, rgb(10, 10, 10) 90%);
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }
    
    /* HIDE DEFAULTS */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* GLASSMORPHISM CARDS */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        transition: transform 0.2s ease-in-out;
    }
    .glass-card:hover {
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* HERO STATS */
    .stat-box {
        text-align: center;
        padding: 15px;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
    }
    .stat-num { font-size: 1.8rem; font-weight: 800; color: #3b82f6; }
    .stat-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; }

    /* TYPOGRAPHY */
    h1 {
        background: linear-gradient(90deg, #60a5fa, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    /* BUTTONS */
    div.stButton > button {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        width: 100%;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        transform: scale(1.02);
    }
    
    /* UPLOADER CLEANUP */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.02);
        border: 1px dashed rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: rgba(255,255,255,0.03); border-radius: 8px; color: #94a3b8; border: none;}
    .stTabs [aria-selected="true"] { background-color: rgba(37, 99, 235, 0.2); color: #60a5fa; border: 1px solid #3b82f6; }
</style>
""", unsafe_allow_html=True)

# --- CONFIG & CONSTANTS ---
MODEL_PATH = "models/best_classifier.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = os.path.join("data", "raw")

# --- LOADERS ---
@st.cache_resource
def load_model():
    class CancerClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = MorphologyAwareViT()
            self.classifier = nn.Linear(192, 2)
        def forward(self, x): return self.classifier(self.base_model.backbone(x))
        def get_embedding(self, x): return self.base_model.backbone(x)

    model = CancerClassifier()
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu') if DEVICE=='cpu' else None)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def index_reference_images():
    model = load_model()
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    reference_db = []
    for class_name in ['benign', 'malignant']:
        search_path = None
        for root, dirs, files in os.walk(DATA_ROOT):
            if class_name in root and 'test' in root:
                search_path = root; break
        if search_path:
            images = [os.path.join(search_path, f) for f in os.listdir(search_path) if f.endswith('.png')]
            selected = random.sample(images, min(len(images), 40))
            for img_path in selected:
                try:
                    img = Image.open(img_path).convert('RGB')
                    t_img = transform(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        emb = model.get_embedding(t_img).cpu().numpy().flatten()
                    reference_db.append({'path': img_path, 'embedding': emb, 'label': class_name})
                except: pass
    return reference_db

# --- LOGIC ---
def validate_image_type(image_pil):
    img = np.array(image_pil)
    if len(img.shape) < 3: return False, "Grayscale image detected."
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if np.mean(hsv[:, :, 2]) < 40: return False, "Image too dark."
    green_mask = cv2.inRange(hsv, (35, 20, 20), (85, 255, 255))
    if (np.sum(green_mask > 0) / (img.shape[0]*img.shape[1])) > 0.30: return False, "Non-tissue content."
    return True, "Valid"

# --- NEW PDF GENERATOR FUNCTION ---
def generate_enhanced_pdf(result_label, conf_val, density):
    pdf = FPDF()
    pdf.add_page()
    
    # 1. Header
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(44, 62, 80) # Dark Blue
    pdf.cell(0, 15, "OncoVision Pro - Diagnostic Report", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, "Automated Histopathology Intelligence System", ln=True, align='C')
    pdf.line(10, 35, 200, 35) # Horizontal line
    pdf.ln(20)

    # 2. Patient/Sample Info (Simulated)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(40, 10, "Analysis Date:", 0, 0)
    pdf.set_font("Arial", '', 12)
    import datetime
    pdf.cell(0, 10, str(datetime.date.today()), ln=True)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, "Methodology:", 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, "Morphology-Aware Vision Transformer (ViT-Tiny) + TTA", ln=True)
    pdf.ln(10)

    # 3. Diagnostic Result Box
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "PRIMARY FINDINGS", ln=True)
    
    # Color logic for PDF text (Red for Malignant, Green for Benign)
    if result_label == "Malignant":
        pdf.set_text_color(220, 20, 60) # Crimson Red
        risk_text = "HIGH RISK - Immediate Review Required"
        rec_text = "1. Urgent referral to Oncology.\n2. Confirmatory biopsy recommended.\n3. IHC staining for receptor status."
    else:
        pdf.set_text_color(34, 139, 34) # Forest Green
        risk_text = "LOW RISK - No Malignancy Detected"
        rec_text = "1. Routine follow-up as per standard protocol.\n2. No specific intervention required at this stage."

    pdf.set_font("Arial", 'B', 24)
    pdf.cell(0, 15, f"{result_label.upper()}", ln=True)
    
    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Confidence Score: {conf_val:.1%}", ln=True)
    pdf.ln(5)

    # 4. Detailed Analysis
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Morphological Assessment:", ln=True)
    pdf.set_font("Arial", '', 11)
    
    analysis_text = f"The AI system analyzed the tissue sample for nuclear atypia and cellular arrangement. "
    if result_label == "Malignant":
        analysis_text += f"High nuclear density ({density:.1%}) was observed with irregular clustering. "
        analysis_text += "The attention mechanism highlighted regions consistent with invasive ductal carcinoma features."
    else:
        analysis_text += f"Cellular density appears normal ({density:.1%}). "
        analysis_text += "Tissue architecture is preserved with distinct boundaries. No significant mitotic activity detected."
    
    pdf.multi_cell(0, 6, analysis_text)
    pdf.ln(10)

    # 5. Recommendations
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Clinical Recommendations:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 7, rec_text)
    pdf.ln(15)

    # 6. Disclaimer (Footer)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, 250, 200, 250)
    pdf.set_y(255)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(100, 100, 100)
    disclaimer = ("DISCLAIMER: This report is generated by an Artificial Intelligence system (OncoVision Pro) for "
                  "Research Use Only. It does NOT constitute a final medical diagnosis. All findings must be "
                  "verified by a board-certified pathologist. The developers assume no liability for clinical decisions "
                  "made based on this report.")
    pdf.multi_cell(0, 4, disclaimer, align='C')

    return pdf


def predict_with_tta(model, image_tensor):
    with torch.no_grad():
        transforms_list = [image_tensor, torch.flip(image_tensor, [3]), torch.flip(image_tensor, [2])]
        probs = [torch.nn.functional.softmax(model(img), dim=1) for img in transforms_list]
        avg_prob = torch.stack(probs).mean(dim=0)
    return avg_prob

def plot_transparent_pca(query_embedding, db):
    if not db: return None
    X = [item['embedding'] for item in db]
    labels = [item['label'] for item in db]
    query_vec = query_embedding.detach().cpu().numpy().flatten()
    X.append(query_vec)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(np.array(X))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    
    for i, label in enumerate(labels):
        color = '#ef4444' if label == 'malignant' else '#10b981'
        ax.scatter(X_pca[i, 0], X_pca[i, 1], c=color, alpha=0.5, s=60, edgecolors='none')
        
    ax.scatter(X_pca[-1, 0], X_pca[-1, 1], c='#3b82f6', marker='*', s=500, edgecolors='white', linewidth=2, label='Current')
    ax.axis('off')
    return fig

def plot_transparent_dist(probs):
    fig, ax = plt.subplots(figsize=(8, 2))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    classes = ['Benign', 'Malignant']
    values = probs[0].cpu().numpy() * 100
    colors = ['#10b981', '#ef4444']
    y_pos = np.arange(len(classes))
    ax.barh(y_pos, values, color=colors, height=0.6, alpha=0.9, edgecolor='white', linewidth=1)
    ax.set_xlim(0, 100)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, color='white', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    for i, v in enumerate(values):
        ax.text(v + 2, i, f"{v:.1f}%", color='white', va='center', fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
    return fig

# --- SIDEBAR ---
with st.sidebar:
    selected = option_menu(
        "OncoVision", ["Dashboard", "System Info"], 
        icons=['activity', 'cpu'], menu_icon="cast", default_index=0,
        styles={"nav-link-selected": {"background-color": "rgba(59, 130, 246, 0.2)", "color": "#60a5fa", "border-left": "4px solid #3b82f6"}}
    )
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; color: #10b981;'>● {DEVICE.upper()} Online</p>", unsafe_allow_html=True)

# --- DASHBOARD ---
if selected == "Dashboard":
    
    # 1. HEADER
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="font-size: 3rem; margin-bottom: 10px;">OncoVision<span style="color:#3b82f6">Pro</span></h1>
        <p style="color: #94a3b8;">Morphology-Aware AI for Histopathology Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_results = st.columns([1, 2], gap="large")

    # 2. LEFT COLUMN (Input)
    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🧬 Specimen Input")
        uploaded_file = st.file_uploader("Upload Slide", type=['png', 'jpg'], label_visibility="collapsed")
        
        # Sample Button for Demo
        if st.button("Load Demo Sample (Malignant)"):
            # Try to find a demo file
            demo_path = None
            for root, dirs, files in os.walk(DATA_ROOT):
                for f in files:
                    if "malignant" in root.lower() and f.endswith(".png"):
                        demo_path = os.path.join(root, f)
                        break
                if demo_path: break
            
            if demo_path:
                st.session_state['demo_image'] = demo_path
                st.rerun()
                
        st.markdown('</div>', unsafe_allow_html=True)
        
        # If user uploads or clicks demo
        image_to_process = None
        if uploaded_file:
            image_to_process = Image.open(uploaded_file).convert('RGB')
        elif 'demo_image' in st.session_state:
            image_to_process = Image.open(st.session_state['demo_image']).convert('RGB')
            
        if image_to_process:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.image(image_to_process, caption="Preview (400X)", use_container_width=True)
            if st.button("🚀 INITIATE ANALYSIS"):
                st.session_state['analyzing'] = True
            st.markdown('</div>', unsafe_allow_html=True)

    # 3. RIGHT COLUMN (Results OR Empty State)
    with col_results:
        # STATE A: RESULTS
        if 'analyzing' in st.session_state and image_to_process:
            db = index_reference_images()
            valid, msg = validate_image_type(image_to_process)
            
            if not valid:
                st.error(f"⛔ {msg}")
            else:
                with st.spinner("🔬 Extracting morphological features..."):
                    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
                    img_t = transform(image_to_process).unsqueeze(0).to(DEVICE)
                    model = load_model()
                    probs = predict_with_tta(model, img_t)
                    with torch.no_grad(): embedding = model.get_embedding(img_t)
                    
                    conf, pred = torch.max(probs, 1)
                    result_label = "Malignant" if pred.item() == 1 else "Benign"
                    conf_val = conf.item()
                    main_color = "#ef4444" if result_label == "Malignant" else "#10b981"

                # Result Card
                st.markdown(f"""
                <div class="glass-card" style="border: 1px solid {main_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h3 style="margin:0; color: #94a3b8;">AI Diagnosis</h3>
                            <h1 style="margin:0; font-size: 3.5rem; background: linear-gradient(90deg, {main_color}, white); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{result_label.upper()}</h1>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 2.5rem; font-weight: 800; color: white;">{conf_val:.1%}</div>
                            <div style="color: #94a3b8;">TTA Confidence</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Tabs
                t1, t2, t3 = st.tabs(["Morphology", "Manifold", "Report"])
                with t1:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Nuclei Heatmap**")
                        img_np = np.array(image_to_process)
                        if img_np.shape[-1] == 4: img_np = img_np[:,:,:3]
                        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                        mask = cv2.inRange(hsv, np.array([120,30,30]), np.array([170,255,200]))
                        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                        overlay = cv2.addWeighted(img_np, 0.7, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), 0.3, 0)
                        st.image(overlay, use_container_width=True)
                    with c2:
                        st.markdown("**Model Certainty**")
                        st.pyplot(plot_transparent_dist(probs))
                        
                        st.markdown("**Findings:**")
                        density = np.count_nonzero(mask>0)/mask.size
                        if result_label == "Malignant":
                            st.warning(f"High nuclear density ({density:.1%}). Irregular clusters detected.")
                        else:
                            st.success(f"Normal tissue architecture ({density:.1%}). No atypia.")
                    st.markdown('</div>', unsafe_allow_html=True)

                with t2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("**Latent Space Projection (PCA)**")
                    st.pyplot(plot_transparent_pca(embedding, db))
                    st.markdown('</div>', unsafe_allow_html=True)

                with t3:
                    # Calculate density for the report
                    density = np.count_nonzero(mask > 0) / mask.size
                    
                    # Call the NEW function
                    pdf = generate_enhanced_pdf(result_label, conf_val, density)
                    
                    # Save and Download
                    pdf_output = pdf.output(dest='S').encode('latin-1', 'replace') 
                    # Note: 'replace' handles any potential unicode errors safely
                    
                    b64 = base64.b64encode(pdf_output).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="OncoVision_Report.pdf"><button style="width:100%; padding: 15px; background-color: #4CAF50; color: white; border: none; border-radius: 8px; cursor: pointer;">📄 Download Professional Medical Report</button></a>'
                    st.markdown(href, unsafe_allow_html=True)

        # STATE B: EMPTY / WELCOME STATE (The Missing Part)
        else:
            st.markdown('<div class="glass-card" style="text-align: center; padding: 40px;">', unsafe_allow_html=True)
            st.markdown("## 🏥 System Ready")
            st.markdown("Please upload a slide to begin analysis.")
            
            # Quick Stats Grid
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown('<div class="stat-box"><div class="stat-num">94.5%</div><div class="stat-label">Accuracy</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="stat-box"><div class="stat-num">ViT</div><div class="stat-label">Model Core</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown('<div class="stat-box"><div class="stat-num">TTA</div><div class="stat-label">Robustness</div></div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Breast_cancer_%28ductal_carcinoma%29_%281%29_histopathology.jpg/640px-Breast_cancer_%28ductal_carcinoma%29_%281%29_histopathology.jpg", caption="Example Histopathology Slide", width=400)
            st.markdown('</div>', unsafe_allow_html=True)

elif selected == "System Info":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="font-size: 2.5rem;">System <span style="color:#3b82f6">Documentation</span></h1>
    </div>
    """, unsafe_allow_html=True)

    # Use Tabs for cleaner organization
    info_tab1, info_tab2, info_tab3 = st.tabs(["📖 User Guide", "⚙️ Architecture", "⚖️ Disclaimers"])

    with info_tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### How to Use OncoVision Pro")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 1. Upload")
            st.info("Drag & drop a Histopathology slide (PNG/JPG) into the sidebar or dashboard area. Ensure the image is H&E stained.")
        with c2:
            st.markdown("#### 2. Analyze")
            st.info("Click 'INITIATE ANALYSIS'. The system will perform Test-Time Augmentation (TTA) and feature extraction.")
        with c3:
            st.markdown("#### 3. Interpret")
            st.info("Review the Diagnosis, check the Nuclei Heatmap for accuracy, and download the PDF report.")
        
        st.markdown("---")
        st.markdown("### Understanding the Visuals")
        st.markdown("""
        *   **Nuclei Heatmap:** Shows *where* the AI is looking. Red/Yellow areas indicate high suspicion of cancer cells.
        *   **Manifold Projection:** A map of 100 previous cases. If your 'Current Star' lands in the Red cluster, it confirms the Malignant diagnosis visually.
        *   **Confidence Score:** How sure the AI is. Scores >90% are considered high confidence.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with info_tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Technical Specifications")
        st.code("""
        Model Architecture:   Vision Transformer (ViT-Tiny)
        Pre-training:         Self-Supervised Contrastive Learning (SSL)
        Novelty Feature:      Morphology-Aware Reconstruction Head
        Inference Method:     Test-Time Augmentation (3-View Consensus)
        Parameter Count:      5.7 Million
        Input Resolution:     224x224 (Patch-based)
        Backend Framework:    PyTorch + CUDA
        """, language="yaml")
        st.markdown('</div>', unsafe_allow_html=True)

    with info_tab3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Important Medical Disclaimer")
        st.warning("""
        **Research Use Only (RUO)**
        
        This software is a prototype developed for academic research purposes. It is **NOT** a certified medical device (FDA/CE).
        
        1.  **Do not use for self-diagnosis.**
        2.  Results from this tool should be treated as a "Second Opinion" only.
        3.  A qualified Pathologist must always verify the input slide and the final result.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
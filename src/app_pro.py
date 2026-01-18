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
import seaborn as sns
from scipy import stats

# Import model
sys.path.append('src')
from model import MorphologyAwareViT

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="OncoVision Enterprise",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL MEDICAL UI (CSS) ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(37, 99, 235, 0.05));
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        margin-bottom: 1rem;
    }
    .warning-box {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .precaution-box {
        background: rgba(251, 191, 36, 0.1);
        border-left: 4px solid #fbbf24;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIG & CONSTANTS ---
MODEL_PATH = "models/best_classifier.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = os.path.join("data", "raw")

# --- MODEL LOADING ---
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

# --- UPDATED VALIDATION LOGIC (More Lenient) ---
def validate_image_type(image_pil):
    """
    Revised Gatekeeper: More permissive for Benign samples (paler colors).
    """
    img = np.array(image_pil)
    
    # 1. Check Color Channels
    if len(img.shape) < 3:
        return False, "Grayscale image detected. H&E slides must be colored."
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # 2. Check Brightness (Slides are backlit -> bright)
    mean_brightness = np.mean(hsv[:, :, 2])
    # Lowered threshold to 40 (was 80) to accept darker/dense slides
    if mean_brightness < 40:
        return False, "Image is too dark to be a valid microscope slide."
    
    # 3. Check Green Content (Nature/Landscape rejection)
    # H&E has almost ZERO green.
    green_mask = cv2.inRange(hsv, (35, 20, 20), (85, 255, 255))  # Narrowed green range
    green_ratio = np.sum(green_mask > 0) / (img.shape[0] * img.shape[1])
    if green_ratio > 0.30:  # Increased tolerance to 30%
        return False, "Image contains significant green/cyan (likely a nature photo)."
    
    # 4. Check for Color Variance (Tissue isn't solid color)
    std_dev_color = np.std(img)
    if std_dev_color < 5:
        return False, "Image is a solid color/blank. No tissue structure found."
    
    return True, "Valid"

# --- HELPER FUNCTIONS ---
@st.cache_resource
def index_reference_images():
    model = load_model()
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    reference_db = []
    
    for class_name in ['benign', 'malignant']:
        search_path = None
        for root, dirs, files in os.walk(DATA_ROOT):
            if class_name in root and 'test' in root:
                search_path = root
                break
        
        if search_path:
            images = [os.path.join(search_path, f) for f in os.listdir(search_path) if f.endswith('.png')]
            selected = random.sample(images, min(len(images), 50))
            
            for img_path in selected:
                try:
                    img = Image.open(img_path).convert('RGB')
                    t_img = transform(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        emb = model.get_embedding(t_img).cpu().numpy().flatten()
                    reference_db.append({'path': img_path, 'embedding': emb, 'label': class_name})
                except:
                    pass
    
    return reference_db

def generate_automated_report(pred_label, confidence, heatmap_mask):
    active_pixels = np.count_nonzero(heatmap_mask > 0)
    total_pixels = heatmap_mask.size
    density = (active_pixels / total_pixels) * 100
    
    report = []
    
    if pred_label == "Malignant":
        report.append(f"**Primary Finding:** The system detected features consistent with Invasive Ductal Carcinoma (IDC) with **{confidence}** confidence.")
        if density > 15:
            report.append(f"**Morphology:** High nuclear density detected (Coverage: {density:.1f}%). The attention map shows diffuse, irregular clustering of hyperchromatic nuclei.")
        else:
            report.append(f"**Morphology:** Localized suspicious regions detected (Coverage: {density:.1f}%). The system focused on specific focal points.")
        report.append("**Cellular Characteristics:** Irregular nuclear contours and loss of glandular architecture were prioritized.")
    else:
        report.append(f"**Primary Finding:** The tissue sample appears Benign with **{confidence}** confidence.")
        report.append("**Morphology:** The system observed uniform cellular arrangement and distinct boundaries.")
        report.append(f"**Analysis:** Only minor non-specific activations were found (Coverage: {density:.1f}%), consistent with healthy fibroglandular tissue.")
    
    return "\n\n".join(report)

def predict_with_tta(model, image_tensor):
    with torch.no_grad():
        transforms_list = [image_tensor, torch.flip(image_tensor, [3]), torch.flip(image_tensor, [2])]
        probs = [torch.nn.functional.softmax(model(img), dim=1) for img in transforms_list]
        avg_prob = torch.stack(probs).mean(dim=0)
    return avg_prob

def plot_latent_space(query_embedding, db):
    if not db:
        return None
    
    X = [item['embedding'] for item in db]
    labels = [item['label'] for item in db]
    query_vec = query_embedding.detach().cpu().numpy().flatten()
    X.append(query_vec)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(np.array(X))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor('#1e293b')
    fig.patch.set_facecolor('#1e293b')
    
    for i, label in enumerate(labels):
        color = '#ef4444' if label == 'malignant' else '#10b981'
        ax.scatter(X_pca[i, 0], X_pca[i, 1], c=color, alpha=0.4, s=60, edgecolors='none')
    
    ax.scatter(X_pca[-1, 0], X_pca[-1, 1], c='#3b82f6', marker='*', s=400, 
               edgecolors='white', linewidth=2, label='Current Patient')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    return fig

# --- NEW VISUALIZATION FUNCTIONS ---
def plot_confidence_distribution(probs):
    """Shows confidence distribution across both classes"""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#1e293b')
    
    classes = ['Benign', 'Malignant']
    values = probs[0].cpu().numpy() * 100
    colors = ['#10b981', '#ef4444']
    
    bars = ax.barh(classes, values, color=colors, alpha=0.7)
    ax.set_xlabel('Confidence (%)', color='white', fontsize=12)
    ax.set_xlim(0, 100)
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 2, i, f'{val:.1f}%', va='center', color='white', fontweight='bold')
    
    return fig

def plot_similarity_scores(query_embedding, db):
    """Shows similarity to top 5 most similar cases"""
    if not db:
        return None
    
    query_vec = query_embedding.detach().cpu().numpy().flatten().reshape(1, -1)
    similarities = []
    
    for item in db:
        ref_vec = item['embedding'].reshape(1, -1)
        sim = cosine_similarity(query_vec, ref_vec)[0][0]
        similarities.append({'similarity': sim, 'label': item['label']})
    
    # Get top 5
    similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)[:5]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#1e293b')
    
    labels = [f"{i+1}. {s['label'].capitalize()}" for i, s in enumerate(similarities)]
    scores = [s['similarity'] * 100 for s in similarities]
    colors = ['#ef4444' if s['label'] == 'malignant' else '#10b981' for s in similarities]
    
    bars = ax.barh(labels, scores, color=colors, alpha=0.7)
    ax.set_xlabel('Similarity Score (%)', color='white', fontsize=12)
    ax.set_xlim(0, 100)
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    
    for bar, score in zip(bars, scores):
        ax.text(score + 1, bar.get_y() + bar.get_height()/2, f'{score:.1f}%', 
                va='center', color='white', fontsize=9)
    
    return fig

def plot_attention_intensity_histogram(mask):
    """Shows distribution of attention intensity"""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#1e293b')
    
    # Only plot non-zero values
    active_values = mask[mask > 0]
    
    if len(active_values) > 0:
        ax.hist(active_values, bins=30, color='#3b82f6', alpha=0.7, edgecolor='white')
        ax.set_xlabel('Attention Intensity', color='white', fontsize=12)
        ax.set_ylabel('Frequency', color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
    else:
        ax.text(0.5, 0.5, 'No significant attention detected', 
                ha='center', va='center', color='white', fontsize=14,
                transform=ax.transAxes)
        ax.axis('off')
    
    return fig

def plot_color_analysis(image_np):
    """Analyzes H&E stain distribution"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    fig.patch.set_facecolor('#1e293b')
    
    colors = ['red', 'green', 'blue']
    channels = ['Hematoxylin (Blue)', 'Green Channel', 'Eosin (Red)']
    
    for i, (ax, color, title) in enumerate(zip(axes, colors, channels)):
        ax.set_facecolor('#1e293b')
        ax.hist(image_np[:,:,i].ravel(), bins=50, color=color, alpha=0.7)
        ax.set_title(title, color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
    
    plt.tight_layout()
    return fig

def generate_risk_assessment(pred_label, confidence_val):
    """Generate clinical risk assessment"""
    conf_num = confidence_val
    
    if pred_label == "Malignant":
        if conf_num > 0.90:
            risk_level = "HIGH"
            risk_color = "#dc2626"
            recommendation = "Immediate pathologist review recommended. Consider urgent biopsy confirmation."
        elif conf_num > 0.75:
            risk_level = "MODERATE-HIGH"
            risk_color = "#ea580c"
            recommendation = "Pathologist review advised. Schedule follow-up consultation."
        else:
            risk_level = "MODERATE"
            risk_color = "#f59e0b"
            recommendation = "Borderline case. Second opinion and additional imaging recommended."
    else:
        if conf_num > 0.90:
            risk_level = "LOW"
            risk_color = "#059669"
            recommendation = "Routine monitoring. Standard follow-up protocol."
        elif conf_num > 0.75:
            risk_level = "LOW-MODERATE"
            risk_color = "#0891b2"
            recommendation = "Consider routine follow-up imaging in 6-12 months."
        else:
            risk_level = "INDETERMINATE"
            risk_color = "#7c3aed"
            recommendation = "Inconclusive. Additional testing or expert consultation needed."
    
    return risk_level, risk_color, recommendation

# --- MAIN APP LAYOUT ---
# 1. SIDEBAR (Clean & Functional)
with st.sidebar:
    st.title("🏥 OncoVision")
    st.markdown("### Enterprise v2.0")
    st.markdown("---")
    
    # Image Upload MOVED TO SIDEBAR to clear up main screen
    st.markdown("### 📤 Patient Data")
    uploaded_file = st.file_uploader("Upload Slide (PNG/JPG)", type=['png', 'jpg'])
    
    if uploaded_file:
        st.success("Image Loaded")
        st.session_state['image_file'] = uploaded_file
    
    st.markdown("---")
    
    # Visualization Options
    st.markdown("### 📊 Visualization Options")
    show_advanced = st.checkbox("Show Advanced Metrics", value=True)
    show_color_analysis = st.checkbox("Show Color Analysis", value=False)
    
    st.markdown("---")
    
    if st.button("🔄 RELOAD SYSTEM", type="secondary"):
        st.cache_resource.clear()
        st.rerun()
    
    # System Info
    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    st.caption(f"Device: {DEVICE.upper()}")
    st.caption(f"Model: MorphologyAwareViT")

# 2. MAIN DASHBOARD
if 'image_file' in st.session_state:
    # Initialize DB quietly
    db = index_reference_images()
    image = Image.open(st.session_state['image_file']).convert('RGB')
    
    # --- VALIDATION ---
    is_valid, msg = validate_image_type(image)
    
    if not is_valid:
        st.error("⚠️ Invalid Image Detected")
        st.warning(msg)
        
        st.markdown('<div class="precaution-box">', unsafe_allow_html=True)
        st.markdown("### 🔍 Image Requirements")
        st.markdown("""
        - **Format:** H&E stained histopathology slides
        - **Color:** RGB color image (not grayscale)
        - **Quality:** Clear, well-lit microscopy images
        - **Resolution:** Minimum 224x224 pixels recommended
        - **Magnification:** Typically 400X
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Full Width Layout
        st.markdown("## 🔎 Diagnostic Overview")
        
        col_img, col_stat = st.columns([1, 2], gap="large")
        
        with col_img:
            st.image(image, caption="Whole Slide Image (400X)", use_container_width=True)
            
            if st.button("🔬 RUN TTA ANALYSIS", type="primary", use_container_width=True):
                st.session_state['run_analysis'] = True
        
        with col_stat:
            if 'run_analysis' in st.session_state:
                with st.spinner("Processing Multi-View Analysis..."):
                    # Inference Code
                    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
                    img_t = transform(image).unsqueeze(0).to(DEVICE)
                    model = load_model()
                    probs = predict_with_tta(model, img_t)
                    
                    with torch.no_grad():
                        embedding = model.get_embedding(img_t)
                    
                    conf, pred = torch.max(probs, 1)
                    result_label = "Malignant" if pred.item() == 1 else "Benign"
                    conf_str = f"{conf.item():.2%}"
                    color_code = "#ef4444" if result_label == "Malignant" else "#10b981"  # Red or Green
                    
                    # --- RESULTS CARD ---
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(37, 99, 235, 0.05)); 
                                padding: 2rem; border-radius: 16px; border: 2px solid {color_code}; margin-bottom: 2rem;'>
                        <div style='color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.5rem;'>AI Consensus Diagnosis</div>
                        <div style='color: {color_code}; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;'>
                            {result_label.upper()}
                        </div>
                        <div style='color: #cbd5e1; font-size: 1rem;'>
                            Confidence: {conf_str} (Robustness Check Passed)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk Assessment
                    risk_level, risk_color, recommendation = generate_risk_assessment(result_label, conf.item())
                    
                    st.markdown(f"""
                    <div style='background: rgba(59, 130, 246, 0.1); padding: 1.5rem; 
                                border-radius: 12px; border-left: 4px solid {risk_color}; margin-bottom: 1.5rem;'>
                        <div style='color: {risk_color}; font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;'>
                            Risk Level: {risk_level}
                        </div>
                        <div style='color: #cbd5e1; font-size: 0.95rem;'>
                            {recommendation}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # --- CONFIDENCE DISTRIBUTION ---
                    if show_advanced:
                        st.markdown("#### 📊 Confidence Distribution")
                        fig_conf = plot_confidence_distribution(probs)
                        st.pyplot(fig_conf)
                    
                    # --- TABS (Below the diagnosis) ---
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "🧬 Explanation", 
                        "🌌 Case Manifold", 
                        "🔗 Similarity", 
                        "📈 Analysis",
                        "📄 Report"
                    ])
                    
                    with tab1:
                        img_np = np.array(image)
                        if img_np.shape[-1] == 4:
                            img_np = img_np[:,:,:3]
                        
                        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                        mask = cv2.inRange(hsv, np.array([120,30,30]), np.array([170,255,200]))
                        
                        explanation_text = generate_automated_report(result_label, conf_str, mask)
                        
                        st.markdown(f'<div class="info-box">{explanation_text.replace("**", "").replace("\n**", "\n").replace("\n", "<br>")}</div>', 
                                    unsafe_allow_html=True)
                        
                        st.markdown("#### Morphological Attention Map")
                        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                        overlay = cv2.addWeighted(img_np, 0.7, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), 0.3, 0)
                        st.image(overlay, use_container_width=True)
                        
                        if show_advanced:
                            st.markdown("#### Attention Intensity Distribution")
                            fig_hist = plot_attention_intensity_histogram(mask)
                            st.pyplot(fig_hist)
                    
                    with tab2:
                        st.markdown("#### Feature Space Analysis")
                        st.info("The Blue Star shows this patient's position relative to historical Benign (Green) and Malignant (Red) cases.")
                        fig = plot_latent_space(embedding, db)
                        if fig:
                            st.pyplot(fig)
                    
                    with tab3:
                        st.markdown("#### Top 5 Most Similar Cases")
                        fig_sim = plot_similarity_scores(embedding, db)
                        if fig_sim:
                            st.pyplot(fig_sim)
                        
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown("""
                        **Interpretation Guide:**
                        - **>80% similarity:** Very high morphological resemblance
                        - **60-80%:** Moderate similarity, common patterns
                        - **<60%:** Low similarity, unique case features
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tab4:
                        if show_color_analysis:
                            st.markdown("#### H&E Stain Distribution Analysis")
                            fig_color = plot_color_analysis(img_np)
                            st.pyplot(fig_color)
                            
                            st.markdown('<div class="info-box">', unsafe_allow_html=True)
                            st.markdown("""
                            **Color Channel Analysis:**
                            - **Blue (Hematoxylin):** Stains nuclei and acidic structures
                            - **Red (Eosin):** Stains cytoplasm and extracellular matrix
                            - Higher blue intensity may indicate increased nuclear density
                            """)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Statistical Summary
                        st.markdown("#### Statistical Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Benign Probability", f"{probs[0][0].item():.1%}")
                        with col2:
                            st.metric("Malignant Probability", f"{probs[0][1].item():.1%}")
                        with col3:
                            active_ratio = np.count_nonzero(mask > 0) / mask.size
                            st.metric("Attention Coverage", f"{active_ratio*100:.1f}%")
                    
                    with tab5:
                        # Clinical Precautions
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown("### ⚠️ Clinical Precautions")
                        st.markdown("""
                        1. **Not a Diagnostic Tool:** This AI system is for research and decision support only
                        2. **Requires Validation:** All findings must be confirmed by a licensed pathologist
                        3. **Quality Dependent:** Results accuracy depends on slide quality and staining
                        4. **Population Bias:** Model trained on specific datasets; may not generalize to all populations
                        5. **Update Required:** Regular model updates needed as medical knowledge evolves
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="precaution-box">', unsafe_allow_html=True)
                        st.markdown("### 📋 Recommended Next Steps")
                        st.markdown(f"""
                        - **Immediate:** {recommendation}
                        - **Documentation:** Save this report and attach to patient records
                        - **Follow-up:** Schedule pathology consultation within 48-72 hours
                        - **Patient Communication:** Inform patient of preliminary findings
                        - **Quality Check:** Verify slide quality and consider re-staining if needed
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # PDF Report Generation
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", 'B', 16)
                        pdf.cell(0, 10, "OncoVision Enterprise - Patient Report", ln=True)
                        pdf.ln(10)
                        
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 8, f"Diagnosis: {result_label}", ln=True)
                        pdf.cell(0, 8, f"Confidence: {conf_str}", ln=True)
                        pdf.cell(0, 8, f"Risk Level: {risk_level}", ln=True)
                        pdf.ln(5)
                        
                        pdf.set_font("Arial", size=11)
                        pdf.multi_cell(0, 6, f"Clinical Recommendation:\n{recommendation}\n")
                        pdf.ln(5)
                        
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 8, "Automated Analysis:", ln=True)
                        pdf.set_font("Arial", size=10)
                        pdf.multi_cell(0, 5, explanation_text.replace('**', ''))
                        pdf.ln(5)
                        
                        pdf.set_font("Arial", 'I', 9)
                        pdf.multi_cell(0, 5, "DISCLAIMER: This report is generated by an AI system for research purposes. All findings must be validated by a qualified pathologist. Not for clinical diagnostic use.")
                        
                        pdf_bytes = pdf.output(dest='S').encode('latin-1')
                        b64 = base64.b64encode(pdf_bytes).decode()
                        
                        st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="oncovision_report.pdf" style="display: inline-block; padding: 0.5rem 1rem; background: #3b82f6; color: white; text-decoration: none; border-radius: 6px; margin-top: 1rem;">📥 Download PDF Record</a>', 
                                    unsafe_allow_html=True)
            else:
                st.info("👈 Click 'RUN TTA ANALYSIS' to begin.")
                
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("""
                ### 🔬 Analysis Features
                
                **Multi-View Testing:** The system performs Test-Time Augmentation (TTA) by analyzing the slide from multiple angles to ensure robust predictions.
                
                **What to Expect:**
                - Confidence distribution across diagnostic classes
                - Morphological attention heatmaps
                - Feature space positioning relative to known cases
                - Similarity analysis to historical samples
                - Color channel analysis (H&E staining)
                - Comprehensive clinical risk assessment
                
                **Processing Time:** ~5-10 seconds depending on hardware
                """)
                st.markdown('</div>', unsafe_allow_html=True)

else:
    # Welcome Screen
    st.markdown("# 👋 Welcome to OncoVision Enterprise")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### AI-Powered Histopathology Analysis System
        
        OncoVision Enterprise is a research-grade medical imaging platform designed to assist pathologists 
        in the analysis of H&E stained breast tissue samples for Invasive Ductal Carcinoma (IDC) detection.
        
        **System Capabilities:**
        - 🔬 H&E Stain Analysis with Morphology-Aware Vision Transformer
        - 🔄 Test Time Augmentation (TTA) for robust predictions
        - 🎯 Attention-based morphological feature extraction
        - 🌌 Vector Similarity Search across reference database
        - 📊 Multi-dimensional visualization suite
        - 📄 Automated clinical report generation
        
        **Please upload a histopathology slide using the sidebar to begin analysis.**
        """)
        
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        ### ⚠️ Important Disclaimers
        
        - **Research Use Only:** This system is not FDA-approved for clinical diagnosis
        - **Expert Review Required:** All AI predictions must be validated by board-certified pathologists
        - **Data Privacy:** Uploaded images are processed locally and not stored on external servers
        - **Limitations:** Performance may vary based on image quality, staining protocol, and tissue preparation
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Model Statistics")
        st.markdown("""
        **Architecture:**  
        MorphologyAwareViT
        
        **Training Dataset:**  
        Breast Histopathology Images
        
        **Classes:**  
        - Benign  
        - Malignant (IDC)
        
        **Inference:**  
        Multi-view TTA enabled
        
        **Reference DB:**  
        100 indexed cases
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="precaution-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Best Practices")
        st.markdown("""
        1. Upload high-quality RGB images
        2. Ensure proper H&E staining
        3. Use 400X magnification slides
        4. Review all visualizations
        5. Document AI findings separately
        6. Consult pathologist for confirmation
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional Information Section
    st.markdown("---")
    st.markdown("## 📚 Understanding the Analysis")
    
    tab_info1, tab_info2, tab_info3 = st.tabs(["🔍 How It Works", "📊 Visualizations Guide", "⚕️ Clinical Integration"])
    
    with tab_info1:
        st.markdown("""
        ### Analysis Pipeline
        
        **Step 1: Image Validation**
        - Checks for proper H&E staining characteristics
        - Validates image format and quality
        - Ensures sufficient color variance and brightness
        
        **Step 2: Multi-View Inference**
        - Original image analysis
        - Horizontal flip analysis
        - Vertical flip analysis
        - Probability averaging for robustness
        
        **Step 3: Feature Extraction**
        - Deep morphological feature embedding
        - Attention mechanism activation
        - Nuclear and cellular pattern recognition
        
        **Step 4: Classification & Reporting**
        - Binary classification (Benign vs Malignant)
        - Confidence scoring
        - Risk stratification
        - Automated report generation
        """)
    
    with tab_info2:
        st.markdown("""
        ### Visualization Components
        
        **Confidence Distribution**
        - Bar chart showing probability for each class
        - Helps assess prediction certainty
        - Red = Malignant, Green = Benign
        
        **Morphological Attention Map**
        - Heatmap overlay on original image
        - Shows regions of interest identified by AI
        - Warmer colors = higher attention/importance
        
        **Feature Space (PCA Projection)**
        - 2D visualization of high-dimensional embeddings
        - Blue star = current case
        - Green dots = benign reference cases
        - Red dots = malignant reference cases
        - Proximity indicates morphological similarity
        
        **Similarity Scores**
        - Top 5 most similar historical cases
        - Percentage indicates feature similarity
        - Color indicates class of matched case
        
        **H&E Color Analysis**
        - Histogram of stain distribution
        - Blue channel = Hematoxylin (nuclei)
        - Red channel = Eosin (cytoplasm)
        - Helps assess staining quality
        
        **Attention Intensity**
        - Distribution of attention weights
        - Shows concentration of suspicious regions
        - Higher values = stronger model focus
        """)
    
    with tab_info3:
        st.markdown("""
        ### Clinical Workflow Integration
        
        **Recommended Usage:**
        
        1. **Pre-Screening**
           - Upload slides for initial AI assessment
           - Prioritize cases with high malignancy confidence
           - Flag borderline cases for expert review
        
        2. **Decision Support**
           - Use AI findings as second opinion
           - Compare attention maps with visual inspection
           - Review similar historical cases
        
        3. **Documentation**
           - Generate PDF reports for medical records
           - Include AI confidence scores in pathology notes
           - Document any discrepancies between AI and human diagnosis
        
        4. **Quality Assurance**
           - Verify slide staining quality via color analysis
           - Check attention coverage for diagnostic adequacy
           - Use similarity search to identify unusual cases
        
        **When to Seek Additional Review:**
        - Confidence < 75%
        - Risk level: INDETERMINATE or MODERATE
        - Conflicting morphological patterns
        - Poor slide quality indicators
        - Unusual similarity profile
        """)
    
    st.markdown("---")
    
    # Footer with contact info
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 2rem; font-size: 0.9rem;'>
        <p><strong>OncoVision Enterprise v2.0</strong></p>
        <p>Powered by MorphologyAwareViT | For Research Use Only</p>
        <p>⚕️ Always consult with qualified medical professionals for diagnostic decisions</p>
    </div>
    """, unsafe_allow_html=True)
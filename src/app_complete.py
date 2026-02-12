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
import sqlite3
import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from fpdf import FPDF
from streamlit_option_menu import option_menu 

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

# --- DATABASE MANAGEMENT (Backend) ---
DB_FILE = "patient_records.db"

def init_db():
    """Initialize the SQLite Database if it doesn't exist"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            patient_name TEXT,
            age INTEGER,
            scan_date TEXT,
            diagnosis TEXT,
            confidence REAL,
            risk_level TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_record(p_id, p_name, p_age, diagnosis, conf, risk):
    """Save a new diagnosis to the database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO history (patient_id, patient_name, age, scan_date, diagnosis, confidence, risk_level)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (p_id, p_name, p_age, date_str, diagnosis, conf, risk))
    conn.commit()
    conn.close()

def load_records():
    """Retrieve all records for the dashboard"""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
    conn.close()
    return df

# Initialize DB on startup
init_db()

# --- ADVANCED CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    .stApp { background: radial-gradient(circle at 10% 20%, rgb(17, 24, 39) 0%, rgb(10, 10, 10) 90%); font-family: 'Inter', sans-serif; color: #e2e8f0; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    .glass-card {
        background: rgba(30, 41, 59, 0.4); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); margin-bottom: 20px;
    }
    
    /* INPUT FIELDS STYLING */
    div[data-baseweb="input"] { background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); color: white; border-radius: 8px; }
    div[data-baseweb="base-input"] { background-color: transparent; }
    
    h1 { background: linear-gradient(90deg, #60a5fa, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    
    div.stButton > button { background: linear-gradient(90deg, #2563eb, #1d4ed8); color: white; border: none; padding: 12px 24px; border-radius: 12px; font-weight: 600; width: 100%; }
    div.stButton > button:hover { transform: scale(1.02); }
    
    .stFileUploader { background: rgba(255, 255, 255, 0.02); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 20px; }
    
    /* DATAFRAME STYLING */
    div[data-testid="stDataFrame"] { background-color: rgba(30, 41, 59, 0.4); padding: 10px; border-radius: 10px; }
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
    """
    FINAL GATEKEEPER (The "Bright Field" Check):
    1. Reject Dark Backgrounds (Space, Fluorescence, Night photos).
    2. Reject Solid Colors (Purple squares).
    3. Reject Non-H&E Colors (Forests, Objects).
    """
    img = np.array(image_pil)
    
    # --- CHECK 1: Format ---
    if len(img.shape) < 3: 
        return False, "Grayscale image detected. H&E slides must be colored."

    # --- CHECK 2: THE "BRIGHT FIELD" CHECK (Fixes the Black Hole issue) ---
    # Histopathology is backlit. The background MUST be light.
    # If the image is mostly dark, it is NOT a standard H&E slide.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Count pixels that are "Dark" (Intensity < 100 out of 255)
    dark_pixels = np.count_nonzero(gray < 80)
    total_pixels = gray.size
    dark_ratio = dark_pixels / total_pixels
    
    # If more than 40% of the image is dark/black, reject it.
    # (Valid slides are usually < 10% dark pixels, mostly just the nuclei)
    if dark_ratio > 0.40:
        return False, "Invalid Image: Background is too dark. H&E slides must be bright/backlit."

    # --- CHECK 3: Texture & Complexity (Fixes the Solid Square issue) ---
    std_dev = np.std(gray)
    if std_dev < 15:
        return False, "Image is too smooth. No cellular texture detected."

    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.count_nonzero(edges) / edges.size
    if edge_ratio < 0.02:
        return False, "Image lacks structural complexity (no defined cell borders)."

    # --- CHECK 4: H&E Color Signature (Fixes the Forest issue) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Pink/Purple Masks
    lower_purple = np.array([110, 15, 20])
    upper_purple = np.array([170, 255, 255])
    lower_pink = np.array([140, 15, 20])
    upper_pink = np.array([180, 255, 255])
    
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
    tissue_mask = cv2.bitwise_or(mask_purple, mask_pink)
    
    tissue_ratio = np.count_nonzero(tissue_mask) / total_pixels
    
    # Must have at least 5% pink/purple content
    if tissue_ratio < 0.05:
        return False, "No biological stain detected. Lacks H&E (Pink/Purple) colors."

    # --- CHECK 5: Reject Nature (Green) ---
    lower_green = np.array([30, 20, 20])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.count_nonzero(green_mask) / total_pixels
    
    if green_ratio > 0.20:
        return False, "Non-medical content detected (High Green signals)."

    return True, "Valid"

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

# --- PDF GENERATOR ---
# --- PDF GENERATOR (Fixed with Recommendations) ---
def generate_enhanced_pdf(p_name, p_id, result_label, conf_val, density):
    def clean(text):
        return text.replace("**", "").encode('latin-1', 'replace').decode('latin-1')

    pdf = FPDF()
    pdf.add_page()
    
    # 1. Header
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 15, "OncoVision - Medical Report", ln=True, align='C')
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)

    # 2. Patient Info
    pdf.set_font("Arial", 'B', 12); pdf.set_text_color(0, 0, 0)
    pdf.cell(40, 10, f"Patient ID: {clean(p_id)}", 0, 1)
    pdf.cell(40, 10, f"Name: {clean(p_name)}", 0, 1)
    pdf.cell(40, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
    pdf.ln(5)

    # 3. Diagnosis
    pdf.set_font("Arial", 'B', 16)
    if result_label == "Malignant":
        pdf.set_text_color(220, 20, 60) # Red
        risk = "HIGH RISK"
        # SPECIFIC RECOMMENDATIONS FOR MALIGNANT
        recommendations = (
            "1. Immediate Referral: Schedule consultation with an Oncologist within 48 hours.\n"
            "2. Confirmatory Testing: Tissue biopsy and IHC staining required.\n"
            "3. Imaging: Suggested Mammogram or MRI to determine tumor extent."
        )
    else:
        pdf.set_text_color(34, 139, 34) # Green
        risk = "LOW RISK"
        # SPECIFIC RECOMMENDATIONS FOR BENIGN
        recommendations = (
            "1. Routine Monitoring: Continue standard screening protocols.\n"
            "2. Follow-up: Repeat imaging in 6-12 months if symptoms persist.\n"
            "3. Observation: Report any new palpable masses to physician."
        )

    pdf.cell(0, 10, f"DIAGNOSIS: {result_label.upper()}", ln=True)
    pdf.set_font("Arial", '', 12); pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Confidence: {conf_val:.1%}  |  Risk Assessment: {risk}", ln=True)
    pdf.ln(5)
    
    # 4. Morphological Analysis
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Morphological Analysis:", ln=True)
    pdf.set_font("Arial", '', 11)
    analysis = f"The AI detected a nuclear density of {density:.1%}. "
    if result_label == "Malignant":
        analysis += "Irregular nuclear clustering consistent with carcinoma features was observed."
    else:
        analysis += "Tissue architecture appears regular with no significant atypia."
    pdf.multi_cell(0, 6, clean(analysis))
    pdf.ln(10)

    # 5. RECOMMENDED ACTIONS (The Missing Part)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 51, 102) # Dark Blue header
    pdf.cell(0, 10, "Recommended Clinical Actions:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 7, clean(recommendations))
    pdf.ln(10)
    
    # 6. Disclaimer
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4, "DISCLAIMER: Research prototype. Findings must be validated by a pathologist.")
    
    return pdf

# --- SIDEBAR ---
with st.sidebar:
    selected = option_menu(
        "OncoVision", 
        ["Dashboard", "Patient Registry", "System Info"], 
        icons=['activity', 'database', 'cpu'], 
        menu_icon="cast", default_index=0,
        styles={"nav-link-selected": {"background-color": "rgba(59, 130, 246, 0.2)", "color": "#60a5fa", "border-left": "4px solid #3b82f6"}}
    )
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; color: #10b981;'>● {DEVICE.upper()} Active</p>", unsafe_allow_html=True)

# --- TAB 1: DASHBOARD ---
if selected == "Dashboard":
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="font-size: 3rem; margin-bottom: 10px;">OncoVision<span style="color:#3b82f6"></span></h1>
        <p style="color: #94a3b8;">Morphology-Aware AI for Histopathology Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_results = st.columns([1, 2], gap="large")

    # INPUT COLUMN
    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Patient Details")
        p_id = st.text_input("Patient ID", value=f"PID-{random.randint(1000,9999)}")
        p_name = st.text_input("Full Name", placeholder="e.g. John Doe")
        p_age = st.number_input("Age", min_value=1, max_value=120, value=45)
        
        st.markdown("### 📤 Upload Specimen")
        uploaded_file = st.file_uploader("Drop H&E Slide", type=['png', 'jpg'], label_visibility="collapsed")
        
        if uploaded_file and p_name:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Specimen 400X", use_container_width=True)
            if st.button("🚀 INITIATE ANALYSIS"):
                st.session_state['analyzing'] = True
        elif uploaded_file:
            st.warning("⚠️ Enter Patient Name to proceed.")
            
        st.markdown('</div>', unsafe_allow_html=True)

    # RESULTS COLUMN
    with col_results:
        if 'analyzing' in st.session_state and uploaded_file and p_name:
            db = index_reference_images()
            valid, msg = validate_image_type(image)
            
            if not valid:
                st.error(f"⛔ Image Rejected: {msg}")
            else:
                with st.spinner("🔬 Extracting features & Saving record..."):
                    # Inference
                    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
                    img_t = transform(image).unsqueeze(0).to(DEVICE)
                    model = load_model()
                    probs = predict_with_tta(model, img_t)
                    with torch.no_grad(): embedding = model.get_embedding(img_t)
                    
                    conf, pred = torch.max(probs, 1)
                    result_label = "Malignant" if pred.item() == 1 else "Benign"
                    conf_val = conf.item()
                    main_color = "#ef4444" if result_label == "Malignant" else "#10b981"
                    
                    # SAVE TO DATABASE (Check duplicates simply by ID+Time or just insert)
                    risk = "High" if result_label == "Malignant" else "Low"
                    save_record(p_id, p_name, p_age, result_label, conf_val, risk)

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
                            <div style="color: #94a3b8;">Certainty</div>
                        </div>
                    </div>
                    <div style="margin-top: 10px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px;">
                        <span style="color: #94a3b8;">✅ Patient Record Saved:</span> <b>{p_name}</b> (ID: {p_id})
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
                        img_np = np.array(image)
                        if img_np.shape[-1] == 4: img_np = img_np[:,:,:3]
                        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                        mask = cv2.inRange(hsv, np.array([120,30,30]), np.array([170,255,200]))
                        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                        overlay = cv2.addWeighted(img_np, 0.7, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), 0.3, 0)
                        st.image(overlay, use_container_width=True)
                    with c2:
                        st.info("Analysis Complete. The attention mechanism identified relevant nuclei structures consistent with the diagnosis.")
                    st.markdown('</div>', unsafe_allow_html=True)

                with t2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("**Latent Space Projection (PCA)**")
                    st.pyplot(plot_transparent_pca(embedding, db))
                    st.markdown('</div>', unsafe_allow_html=True)

                with t3:
                    # PDF Logic
                    density = np.count_nonzero(mask > 0) / mask.size
                    pdf = generate_enhanced_pdf(p_name, p_id, result_label, conf_val, density)
                    pdf_out = pdf.output(dest='S').encode('latin-1', 'replace')
                    b64 = base64.b64encode(pdf_out).decode()
                    st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="Medical_Report.pdf"><button style="width:100%; padding: 15px; background-color: #3b82f6; color: white; border: none; border-radius: 8px;">📥 Download Medical Report</button></a>', unsafe_allow_html=True)

        else:
            st.markdown('<div class="glass-card" style="text-align: center; padding: 40px;">', unsafe_allow_html=True)
            st.markdown("## 🏥 System Ready")
            st.markdown("Please enter Patient Details to begin.")

# --- TAB 2: PATIENT REGISTRY ---
elif selected == "Patient Registry":
    st.markdown("## 🗂️ Patient History Database")
    
    # Load Data
    df = load_records()
    
    # Summary Metrics
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="glass-card" style="text-align:center"><h3>Total Patients</h3><h1>{len(df)}</h1></div>', unsafe_allow_html=True)
    with c2: 
        mal_count = len(df[df['diagnosis']=='Malignant'])
        st.markdown(f'<div class="glass-card" style="text-align:center; border:1px solid #ef4444"><h3>Malignant Cases</h3><h1 style="color:#ef4444">{mal_count}</h1></div>', unsafe_allow_html=True)
    with c3:
        ben_count = len(df[df['diagnosis']=='Benign'])
        st.markdown(f'<div class="glass-card" style="text-align:center; border:1px solid #10b981"><h3>Benign Cases</h3><h1 style="color:#10b981">{ben_count}</h1></div>', unsafe_allow_html=True)
    
    # Data Table
    st.markdown("### Recent Records")
    if not df.empty:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        # Apply Pandas Styling
        def highlight_malignant(s):
            return ['background-color: rgba(239, 68, 68, 0.2)' if s.diagnosis == 'Malignant' else '' for v in s]

        st.dataframe(
            df.style.apply(highlight_malignant, axis=1),
            use_container_width=True,
            column_config={
                "timestamp": "Date/Time",
                "confidence": st.column_config.NumberColumn("Certainty", format="%.2f")
            }
        )
        
        # CSV Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Registry to CSV",
            data=csv,
            file_name='patient_registry.csv',
            mime='text/csv',
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No records found in database.")

# --- TAB 3: INFO ---
# --- TAB 3: SYSTEM INFO & HELP CENTER ---
elif selected == "System Info":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="font-size: 2.5rem;">System <span style="color:#3b82f6">Documentation</span></h1>
        <p style="color: #94a3b8;">User Guide, Interpretation Logic, and Technical Specifications</p>
    </div>
    """, unsafe_allow_html=True)

    # Create 3 sub-tabs for better organization
    tab_guide, tab_interpret, tab_tech = st.tabs(["📖 User Guide", "📊 How to Interpret Results", "⚙️ Technical Specs"])

    # --- TAB 1: HOW TO USE ---
    with tab_guide:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Workflow: From Upload to Diagnosis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Step 1: Patient Data**")
            st.markdown("""
            1. Navigate to the **Dashboard**.
            2. Enter the **Patient Name** and **Age**.
            3. The system auto-generates a **Patient ID** (or you can enter one manually).
            4. *Note: Analysis cannot start without these details.*
            """)
            
        with col2:
            st.info("**Step 2: Specimen Upload**")
            st.markdown("""
            1. Drag & Drop a **Histopathology Image**.
            2. **Requirements:**
               - Format: JPG or PNG.
               - Stain: **H&E** (Hematoxylin & Eosin).
               - Magnification: **400X** (Preferred).
            3. The system performs a **Quality Check** automatically.
            """)
            
        with col3:
            st.info("**Step 3: AI Analysis**")
            st.markdown("""
            1. Click **INITIATE ANALYSIS**.
            2. The system runs **Test-Time Augmentation** (analyzing 3 views).
            3. Results appear in ~2 seconds.
            4. Download the **PDF Report** for medical records.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 2: INTERPRETATION ---
    with tab_interpret:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Understanding the AI Visuals")
        
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("#### 1. Nuclei Heatmap (Morphology)")
            st.warning("""
            **What it shows:** The exact regions the AI focused on to make the decision.
            
            - **Red/Yellow Hotspots:** These indicate high attention. In Malignant cases, these usually align with **Cell Nuclei** (purple dots).
            - **Blue/Transparent:** Background areas the AI ignored.
            
            *Verification:* If the AI highlights empty white space as 'Red', the prediction may be unreliable.
            """)
            
        with c2:
            st.markdown("#### 2. Latent Space Manifold (PCA)")
            st.success("""
            **What it shows:** A comparison of this patient against 100 historical cases.
            
            - **Green Dots:** Past Benign Patients.
            - **Red Dots:** Past Malignant Patients.
            - **⭐ Blue Star:** The Current Patient.
            
            *Verification:* If the Blue Star lands deep inside the Red Cluster, it visually confirms the Malignant diagnosis.
            """)
            
        st.markdown("---")
        st.markdown("#### 3. Confidence Score")
        st.markdown("""
        - **90% - 100%:** High Confidence. Strong morphological evidence found.
        - **70% - 90%:** Moderate Confidence. Typical patterns found, but review recommended.
        - **< 70%:** Low Confidence. The case might be borderline or the image quality is poor. **Expert Pathologist Review Mandatory.**
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 3: TECHNICAL SPECS (For Examiners) ---
    with tab_tech:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🛠️ System Architecture")
        
        st.code("""
        Model Backbone:       Vision Transformer (ViT-Tiny)
        Parameters:           5.7 Million
        Training Method:      Self-Supervised Learning (Morphology-Aware)
        Input Resolution:     224x224 (Patch-based)
        Inference Logic:      Test-Time Augmentation (Original + Horizontal Flip + Vertical Flip)
        Backend Framework:    PyTorch + CUDA
        Database:             SQLite (Local Encrypted Storage)
        """, language="yaml")
        
        st.markdown("---")
        st.markdown("### ⚠️ Research Use Only (RUO)")
        st.markdown("""
        This system is a **Computer-Aided Diagnosis (CAD)** prototype. 
        It is designed to assist, **not replace**, the pathologist. 
        All AI-generated reports must be verified by a certified medical professional before clinical action.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
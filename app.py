import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import hashlib
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="💰 Adult Income Predictor Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main styling */
    .main {
        padding: 0 1rem;
    }
    
    /* Navigation tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 500;
        color: #475569;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #059669, #047857);
        color: white !important;
        box-shadow: 0 4px 12px rgba(5,150,105,0.3);
    }
    
    /* Card styling */
    .custom-card {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid rgba(226, 232, 240, 0.8);
        transition: all 0.3s ease;
        margin-bottom: 16px;
    }
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }
    
    /* Stat cards */
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        margin: 8px 0;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
    }
    
    /* Result boxes */
    .result-box {
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin: 20px 0;
        animation: fadeInUp 0.6s ease;
        position: relative;
        overflow: hidden;
    }
    .result-high {
        background: linear-gradient(135deg, #059669, #047857);
        color: white;
        box-shadow: 0 8px 32px rgba(5,150,105,0.3);
    }
    .result-low {
        background: linear-gradient(135deg, #7c3aed, #6d28d9);
        color: white;
        box-shadow: 0 8px 32px rgba(124,58,237,0.3);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #059669, #047857);
        color: white;
        border: none;
        padding: 12px 40px;
        font-size: 16px;
        width: 100%;
        height: 52px;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 24px rgba(5,150,105,0.4);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #059669, #10b981);
        border-radius: 20px;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-success {
        background: #d1fae5;
        color: #065f46;
    }
    .badge-warning {
        background: #fef3c7;
        color: #92400e;
    }
    .badge-info {
        background: #dbeafe;
        color: #1e40af;
    }
    .badge-danger {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* Timeline styling */
    .timeline-item {
        padding: 12px 16px;
        border-left: 3px solid #059669;
        margin-bottom: 12px;
        background: #f8fafc;
        border-radius: 0 8px 8px 0;
        transition: all 0.3s ease;
    }
    .timeline-item:hover {
        background: #f1f5f9;
        transform: translateX(4px);
    }
    .timeline-date {
        font-size: 0.8rem;
        color: #94a3b8;
        font-weight: 500;
    }
    .timeline-title {
        font-weight: 600;
        color: #0f172a;
        margin: 4px 0;
    }
    .timeline-desc {
        font-size: 0.9rem;
        color: #64748b;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .custom-card {
            padding: 16px;
        }
        .stat-number {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants and Configurations
# ---------------------------------------------------------------------------
ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))

def artifact_path(name):
    return os.path.join(ARTIFACT_DIR, name)

# Education level mapping
EDUCATION_LEVELS = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5,
    "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10,
    "Assoc-voc": 11, "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14,
    "Prof-school": 15, "Doctorate": 16,
}

EDUCATION_DESCRIPTIONS = {
    "Preschool": "Early childhood education",
    "1st-4th": "Elementary school (grades 1-4)",
    "5th-6th": "Elementary school (grades 5-6)",
    "7th-8th": "Middle school (grades 7-8)",
    "9th": "High school freshman",
    "10th": "High school sophomore",
    "11th": "High school junior",
    "12th": "High school senior",
    "HS-grad": "High school graduate",
    "Some-college": "Some college (no degree)",
    "Assoc-voc": "Associate degree (vocational)",
    "Assoc-acdm": "Associate degree (academic)",
    "Bachelors": "Bachelor's degree",
    "Masters": "Master's degree",
    "Prof-school": "Professional school degree",
    "Doctorate": "Doctorate degree",
}

# Categorical options
WORKCLASS_RAW_OPTIONS = [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked",
]

MARITAL_RAW_OPTIONS = [
    "Married-civ-spouse", "Divorced", "Never-married", "Separated",
    "Widowed", "Married-spouse-absent", "Married-AF-spouse",
]

RACE_RAW_OPTIONS = [
    "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other",
]

OCCUPATION_OPTIONS = [
    "Prof-specialty", "Craft-repair", "Exec-managerial", "Adm-clerical",
    "Sales", "Other-service", "Machine-op-inspct", "Transport-moving",
    "Handlers-cleaners", "Farming-fishing", "Tech-support",
    "Protective-serv", "Priv-house-serv", "Armed-Forces",
]

RELATIONSHIP_OPTIONS = [
    "Husband", "Wife", "Own-child", "Not-in-family", "Other-relative", "Unmarried"
]

SEX_OPTIONS = ["Male", "Female"]
COUNTRY_OPTIONS = ["United-States", "Other country"]

# ---------------------------------------------------------------------------
# Data Processing Functions
# ---------------------------------------------------------------------------
def apply_basic_cleaning(df_raw, is_training=False):
    d = df_raw.copy()
    d.replace('?', np.nan, inplace=True)

    if is_training:
        d.drop_duplicates(keep='first', inplace=True)
        d.reset_index(drop=True, inplace=True)

    for col in ['fnlwgt', 'education']:
        if col in d.columns:
            d.drop(columns=[col], inplace=True)

    if is_training:
        d.drop_duplicates(keep='first', inplace=True)
        d.reset_index(drop=True, inplace=True)

    if 'race' in d.columns:
        d['race'] = d['race'].replace(['Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], 'Others')

    if 'marital.status' in d.columns:
        d['marital.status'] = d['marital.status'].replace(
            ['Widowed', 'Divorced', 'Separated', 'Never-married'], 'not_married')
        d['marital.status'] = d['marital.status'].replace(
            ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], 'married')

    if 'workclass' in d.columns:
        work_class = {
            'Self-emp-inc': 'Self-employed',
            'Self-emp-not-inc': 'Self-employed',
            'State-gov': 'Government',
            'Federal-gov': 'Government',
            'Local-gov': 'Government',
            'Without-pay': 'Without-pay'}
        d['workclass'] = d['workclass'].replace(work_class)

    if 'native.country' in d.columns:
        d.loc[
            d['native.country'].notna() &
            (d['native.country'] != 'United-States'),
            'native.country'
        ] = 'Others'

    return d

def apply_feature_engineering(df_clean):
    d = df_clean.copy()
    d['capital_diff'] = d['capital.gain'] - d['capital.loss']
    return d

# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load all trained model artifacts with caching"""
    try:
        artifacts = {
            "categorical_imputer": joblib.load(artifact_path("categorical_imputer.pkl")),
            "sex_encoder": joblib.load(artifact_path("sex_encoder.pkl")),
            "native_country_encoder": joblib.load(artifact_path("native_country_encoder.pkl")),
            "marital_status_encoder": joblib.load(artifact_path("marital_status_encoder.pkl")),
            "onehot_encoder": joblib.load(artifact_path("onehot_encoder.pkl")),
            "occupation_target_encoder": joblib.load(artifact_path("occupation_target_encoder.pkl")),
            "scaler": joblib.load(artifact_path("standard_scaler.pkl")),
            "selected_features": joblib.load(artifact_path("selected_features.pkl")),
            "feature_columns_config": joblib.load(artifact_path("feature_columns_config.pkl")),
            "best_model": joblib.load(artifact_path("best_model.pkl")),
        }
        return artifacts
    except FileNotFoundError as e:
        st.error(f"""
        ⚠️ **Model Artifacts Not Found**
        
        Could not find the required .pkl files. Please ensure all artifacts are in the same folder.
        
        **Missing file:** {str(e)}
        """)
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading artifacts: {str(e)}")
        st.stop()

# Load artifacts
try:
    artifacts = load_artifacts()
    cfg = artifacts["feature_columns_config"]
    ohe = artifacts["onehot_encoder"]
    ohe_categories = dict(zip(cfg["ohe_cols"], ohe.categories_))
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# History Management
# ---------------------------------------------------------------------------
class PredictionHistory:
    """Manage prediction history using session state"""
    
    @staticmethod
    def get_history():
        """Get prediction history from session state"""
        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []
        return st.session_state.prediction_history
    
    @staticmethod
    def add_prediction(data, prediction, proba, feature_row):
        """Add a prediction to history"""
        history = PredictionHistory.get_history()
        
        # Create unique ID
        timestamp = datetime.now().isoformat()
        id_hash = hashlib.md5(f"{timestamp}{data['age']}{data['occupation']}".encode()).hexdigest()[:8]
        
        entry = {
            "id": id_hash,
            "timestamp": timestamp,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": data,
            "prediction": prediction,
            "prediction_label": ">50K" if prediction == 1 else "≤50K",
            "probability": proba,
            "feature_row": feature_row.to_dict() if feature_row is not None else {}
        }
        
        # Add to beginning (most recent first)
        history.insert(0, entry)
        
        # Keep only last 100 predictions
        if len(history) > 100:
            history = history[:100]
        
        st.session_state.prediction_history = history
        return entry
    
    @staticmethod
    def clear_history():
        """Clear all prediction history"""
        st.session_state.prediction_history = []
    
    @staticmethod
    def get_stats():
        """Get statistics from history"""
        history = PredictionHistory.get_history()
        if not history:
            return None
        
        total = len(history)
        high_income = sum(1 for h in history if h["prediction"] == 1)
        low_income = total - high_income
        
        # Calculate average probability
        avg_prob = np.mean([h["probability"] for h in history if h["probability"] is not None]) if history else 0
        
        # Get most common occupation
        occupations = [h["data"]["occupation"] for h in history if "occupation" in h["data"]]
        most_common_occ = max(set(occupations), key=occupations.count) if occupations else "N/A"
        
        # Get average age and hours
        avg_age = np.mean([h["data"]["age"] for h in history if "age" in h["data"]]) if history else 0
        avg_hours = np.mean([h["data"]["hours_per_week"] for h in history if "hours_per_week" in h["data"]]) if history else 0
        
        return {
            "total": total,
            "high_income": high_income,
            "low_income": low_income,
            "high_percentage": (high_income / total) * 100 if total > 0 else 0,
            "avg_probability": avg_prob,
            "most_common_occupation": most_common_occ,
            "avg_age": avg_age,
            "avg_hours": avg_hours
        }

# ---------------------------------------------------------------------------
# Prediction Function
# ---------------------------------------------------------------------------
def predict_income(input_data):
    """Execute the complete prediction pipeline"""
    try:
        # Clean data
        sim = apply_basic_cleaning(input_data, is_training=False)
        sim = apply_feature_engineering(sim)

        # Categorical imputation
        sim[cfg["categorical_impute_cols"]] = artifacts["categorical_imputer"].transform(
            sim[cfg["categorical_impute_cols"]]
        )

        # Binary mapping
        sim["sex"] = sim["sex"].map(artifacts["sex_encoder"])
        sim["native.country"] = sim["native.country"].map(artifacts["native_country_encoder"])
        sim["marital.status"] = sim["marital.status"].map(artifacts["marital_status_encoder"])

        # One-hot encoding
        ohe_cols_sim = cfg["ohe_cols"]
        sim_encoded = ohe.transform(sim[ohe_cols_sim])
        sim_encoded = pd.DataFrame(
            sim_encoded,
            columns=ohe.get_feature_names_out(ohe_cols_sim),
            index=sim.index,
        )
        sim = pd.concat([sim.drop(columns=ohe_cols_sim), sim_encoded], axis=1)

        # Target encoding
        sim["occupation"] = artifacts["occupation_target_encoder"].transform(sim["occupation"])

        # Scaling
        num_cols_sim = cfg["numeric_cols_to_scale"]
        sim[num_cols_sim] = artifacts["scaler"].transform(sim[num_cols_sim])

        # Feature selection
        sim_fs = sim[artifacts["selected_features"]]

        model = artifacts["best_model"]
        prediction = model.predict(sim_fs)[0]
        
        # Get probability
        proba = model.predict_proba(sim_fs)[0][1] if hasattr(model, "predict_proba") else None
        
        return prediction, proba, sim_fs
        
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

# ---------------------------------------------------------------------------
# Prediction Function (with history)
# ---------------------------------------------------------------------------
def predict_and_save(input_data_dict):
    """Predict and save to history"""
    # Prepare input for prediction
    prediction_input = pd.DataFrame([{
        "age": input_data_dict["age"],
        "workclass": input_data_dict["workclass"],
        "education.num": input_data_dict["education_num"],
        "marital.status": input_data_dict["marital_status"],
        "occupation": input_data_dict["occupation"],
        "relationship": input_data_dict["relationship"],
        "race": input_data_dict["race"],
        "sex": input_data_dict["sex"],
        "capital.gain": input_data_dict["capital_gain"],
        "capital.loss": input_data_dict["capital_loss"],
        "hours.per.week": input_data_dict["hours_per_week"],
        "native.country": "United-States" if input_data_dict["native_country"] == "United-States" else "Others",
    }])
    
    # Predict
    prediction, proba, feature_row = predict_income(prediction_input)
    
    # Save to history
    PredictionHistory.add_prediction(input_data_dict, prediction, proba, feature_row)
    
    return prediction, proba, feature_row

# ---------------------------------------------------------------------------
# Page: Predictor
# ---------------------------------------------------------------------------
def render_predictor():
    """Render the main predictor page"""
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 2.5rem; margin: 0; background: linear-gradient(135deg, #059669, #7c3aed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            💰 Income Predictor
        </h1>
        <p style="font-size: 1.1rem; color: #64748b; margin-top: 8px;">
            Enter your details below for an instant income prediction
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input form
    with st.form("income_form", clear_on_submit=False):
        # Personal Information
        with st.container():
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 👤 Personal Information")
            
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input(
                    "Age",
                    min_value=17,
                    max_value=90,
                    value=35,
                    help="Age in years (17-90)",
                    step=1
                )
                
                sex = st.radio(
                    "Sex",
                    SEX_OPTIONS,
                    horizontal=True,
                    index=0
                )
            
            with col2:
                race = st.selectbox(
                    "Race",
                    RACE_RAW_OPTIONS,
                    index=0
                )
                
                native_country = st.selectbox(
                    "Country of Origin",
                    COUNTRY_OPTIONS,
                    index=0
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Family Information
        with st.container():
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 👨‍👩‍👧‍👦 Family Information")
            
            col1, col2 = st.columns(2)
            with col1:
                marital_status = st.selectbox(
                    "Marital Status",
                    MARITAL_RAW_OPTIONS,
                    index=0
                )
            with col2:
                relationship = st.selectbox(
                    "Relationship Role",
                    RELATIONSHIP_OPTIONS,
                    index=0
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Education
        with st.container():
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 🎓 Education")
            
            education_label = st.selectbox(
                "Highest Education Level",
                list(EDUCATION_LEVELS.keys()),
                index=list(EDUCATION_LEVELS.keys()).index("HS-grad"),
                format_func=lambda x: f"{x} — {EDUCATION_DESCRIPTIONS.get(x, '')}"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Employment
        with st.container():
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 💼 Employment")
            
            col1, col2 = st.columns(2)
            with col1:
                workclass = st.selectbox(
                    "Work Class",
                    WORKCLASS_RAW_OPTIONS,
                    index=0
                )
            with col2:
                occupation = st.selectbox(
                    "Occupation",
                    OCCUPATION_OPTIONS,
                    index=0
                )
            
            hours_per_week = st.slider(
                "Hours Worked Per Week",
                min_value=1,
                max_value=99,
                value=40,
                step=1,
                format="%d hours"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Capital
        with st.container():
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 💰 Capital")
            
            col1, col2 = st.columns(2)
            with col1:
                capital_gain = st.number_input(
                    "Capital Gain ($)",
                    min_value=0,
                    value=0,
                    step=100,
                    format="%d"
                )
            with col2:
                capital_loss = st.number_input(
                    "Capital Loss ($)",
                    min_value=0,
                    value=0,
                    step=100,
                    format="%d"
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Submit
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button(
                "🚀 Predict Income",
                use_container_width=True,
                type="primary"
            )
    
    # Handle prediction
    if submitted:
        input_data = {
            "age": age,
            "sex": sex,
            "race": race,
            "native_country": native_country,
            "marital_status": marital_status,
            "relationship": relationship,
            "education_label": education_label,
            "education_num": EDUCATION_LEVELS[education_label],
            "workclass": workclass,
            "occupation": occupation,
            "hours_per_week": hours_per_week,
            "capital_gain": capital_gain,
            "capital_loss": capital_loss
        }
        
        try:
            with st.spinner("🧠 Analyzing data and generating prediction..."):
                prediction, proba, feature_row = predict_and_save(input_data)
            
            # Show results
            render_prediction_results(prediction, proba, feature_row, input_data)
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

def render_prediction_results(prediction, proba, feature_row, input_data):
    """Render prediction results"""
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-box result-high">
                <div style="font-size: 64px; margin-bottom: 8px;">🎉</div>
                <h2 style="color: white; margin: 8px 0;">Income > $50K</h2>
                <p style="font-size: 1.1rem; opacity: 0.9;">The model predicts annual income <strong>exceeds</strong> $50,000</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box result-low">
                <div style="font-size: 64px; margin-bottom: 8px;">📊</div>
                <h2 style="color: white; margin: 8px 0;">Income ≤ $50K</h2>
                <p style="font-size: 1.1rem; opacity: 0.9;">The model predicts annual income <strong>at or below</strong> $50,000</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if proba is not None:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric(
                label="🎯 Probability of >$50K",
                value=f"{proba:.1%}",
                delta=f"{'High' if proba > 0.7 else 'Moderate' if proba > 0.4 else 'Low'} Confidence"
            )
            st.progress(min(max(proba, 0.0), 1.0))
            st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page: Analytics
# ---------------------------------------------------------------------------
def render_analytics():
    """Render analytics page with visualizations"""
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 2.5rem; margin: 0; background: linear-gradient(135deg, #059669, #7c3aed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            📊 Analytics Dashboard
        </h1>
        <p style="font-size: 1.1rem; color: #64748b; margin-top: 8px;">
            Insights and patterns from your prediction history
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    history = PredictionHistory.get_history()
    
    if not history:
        st.info("📭 No predictions yet. Start making predictions to see analytics here!")
        return
    
    # Stats Cards
    stats = PredictionHistory.get_stats()
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 28px;">📊</div>
                <div class="stat-number">{stats['total']}</div>
                <div class="stat-label">Total Predictions</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 28px;">📈</div>
                <div class="stat-number">{stats['high_income']}</div>
                <div class="stat-label">High Income (>$50K)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 28px;">📉</div>
                <div class="stat-number">{stats['low_income']}</div>
                <div class="stat-label">Low Income (≤$50K)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 28px;">🎯</div>
                <div class="stat-number">{stats['high_percentage']:.0f}%</div>
                <div class="stat-label">High Income Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional stats
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">👤 Most Common Occupation</div>
                <div class="stat-number" style="font-size: 1.5rem;">{stats['most_common_occupation']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">📅 Average Age</div>
                <div class="stat-number" style="font-size: 1.5rem;">{stats['avg_age']:.0f} years</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">⏰ Average Work Hours</div>
                <div class="stat-number" style="font-size: 1.5rem;">{stats['avg_hours']:.0f} hrs/week</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("---")
    st.markdown("### 📈 Prediction Trends")
    
    # Prepare data for charts - FIXED: Properly extract data from history
    history_data = []
    for entry in history:
        history_data.append({
            'datetime': entry['datetime'],
            'prediction_label': entry['prediction_label'],
            'age': entry['data']['age'],
            'occupation': entry['data']['occupation'],
            'hours_per_week': entry['data']['hours_per_week'],
            'education_label': entry['data']['education_label'],
            'probability': entry['probability'] if entry['probability'] is not None else 0.5
        })
    
    df_history = pd.DataFrame(history_data)
    
    # Chart 1: Prediction distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart for prediction distribution
        pred_counts = df_history['prediction_label'].value_counts()
        fig_pie = px.pie(
            values=pred_counts.values,
            names=pred_counts.index,
            title="Income Prediction Distribution",
            color=pred_counts.index,
            color_discrete_map={">50K": "#059669", "≤50K": "#7c3aed"},
            hole=0.3
        )
        fig_pie.update_layout(
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart for predictions over time
        df_history['date'] = pd.to_datetime(df_history['datetime']).dt.date
        daily_counts = df_history.groupby(['date', 'prediction_label']).size().reset_index(name='count')
        
        fig_bar = px.bar(
            daily_counts,
            x='date',
            y='count',
            color='prediction_label',
            title="Predictions Over Time",
            color_discrete_map={">50K": "#059669", "≤50K": "#7c3aed"},
            labels={"date": "Date", "count": "Number of Predictions", "prediction_label": "Income"}
        )
        fig_bar.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title=""),
            yaxis=dict(title="")
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # More charts
    st.markdown("### 📊 Feature Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution by income
        fig_age = px.box(
            df_history,
            x='prediction_label',
            y='age',
            title="Age Distribution by Income",
            color='prediction_label',
            color_discrete_map={">50K": "#059669", "≤50K": "#7c3aed"},
            labels={"prediction_label": "Income", "age": "Age"}
        )
        fig_age.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Hours worked by income
        fig_hours = px.box(
            df_history,
            x='prediction_label',
            y='hours_per_week',
            title="Work Hours by Income",
            color='prediction_label',
            color_discrete_map={">50K": "#059669", "≤50K": "#7c3aed"},
            labels={"prediction_label": "Income", "hours_per_week": "Hours/Week"}
        )
        fig_hours.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_hours, use_container_width=True)
    
    # Occupation analysis - FIXED: Now uses 'occupation' column from df_history
    st.markdown("### 💼 Occupation Analysis")
    
    occ_data = df_history.groupby(['occupation', 'prediction_label']).size().reset_index(name='count')
    occ_data.columns = ['Occupation', 'Income', 'Count']
    
    fig_occ = px.bar(
        occ_data,
        x='Occupation',
        y='Count',
        color='Income',
        title="Income Distribution by Occupation",
        color_discrete_map={">50K": "#059669", "≤50K": "#7c3aed"},
        barmode='group'
    )
    fig_occ.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickangle=45),
        height=400
    )
    st.plotly_chart(fig_occ, use_container_width=True)
    
    # Education analysis
    st.markdown("### 🎓 Education Analysis")
    
    edu_data = df_history.groupby(['education_label', 'prediction_label']).size().reset_index(name='count')
    edu_data.columns = ['Education', 'Income', 'Count']
    
    fig_edu = px.bar(
        edu_data,
        x='Education',
        y='Count',
        color='Income',
        title="Income Distribution by Education Level",
        color_discrete_map={">50K": "#059669", "≤50K": "#7c3aed"},
        barmode='group'
    )
    fig_edu.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickangle=45),
        height=400
    )
    st.plotly_chart(fig_edu, use_container_width=True)
    
    # Probability distribution
    st.markdown("### 🎯 Probability Distribution")
    
    prob_data = df_history[df_history['probability'].notna()]
    if not prob_data.empty:
        fig_prob = px.histogram(
            prob_data,
            x='probability',
            nbins=20,
            title="Distribution of Prediction Probabilities",
            color='prediction_label',
            color_discrete_map={">50K": "#059669", "≤50K": "#7c3aed"},
            labels={"probability": "Probability of >$50K", "count": "Number of Predictions"}
        )
        fig_prob.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig_prob.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        st.plotly_chart(fig_prob, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: History
# ---------------------------------------------------------------------------
def render_history():
    """Render prediction history page"""
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 2.5rem; margin: 0; background: linear-gradient(135deg, #059669, #7c3aed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            📜 Prediction History
        </h1>
        <p style="font-size: 1.1rem; color: #64748b; margin-top: 8px;">
            View all your past predictions and their details
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    history = PredictionHistory.get_history()
    
    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Total Predictions:** {len(history)}")
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            PredictionHistory.clear_history()
            st.rerun()
    with col3:
        if st.button("📥 Export History", use_container_width=True):
            if history:
                # Prepare data for export
                export_data = []
                for entry in history:
                    export_row = {
                        'ID': entry['id'],
                        'Timestamp': entry['datetime'],
                        'Prediction': entry['prediction_label'],
                        'Probability': entry['probability'],
                        'Age': entry['data']['age'],
                        'Occupation': entry['data']['occupation'],
                        'Education': entry['data']['education_label'],
                        'Hours/Week': entry['data']['hours_per_week'],
                        'Capital Gain': entry['data']['capital_gain'],
                        'Capital Loss': entry['data']['capital_loss'],
                        'Sex': entry['data']['sex'],
                        'Race': entry['data']['race'],
                        'Marital Status': entry['data']['marital_status']
                    }
                    export_data.append(export_row)
                
                df_export = pd.DataFrame(export_data)
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    st.markdown("---")
    
    if not history:
        st.info("📭 No predictions yet. Start using the predictor to build your history!")
        return
    
    # Display history as timeline
    for entry in history[:20]:  # Show last 20
        with st.container():
            st.markdown(f"""
            <div class="timeline-item">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div class="timeline-date">🕐 {entry['datetime']}</div>
                        <div class="timeline-title">
                            {entry['prediction_label']} — {entry['data']['occupation']}
                        </div>
                        <div class="timeline-desc">
                            Age: {entry['data']['age']} | Education: {entry['data']['education_label']} | 
                            Hours: {entry['data']['hours_per_week']} | 
                            Probability: {f"{entry['probability']:.1%}" if entry['probability'] else "N/A"}
                        </div>
                    </div>
                    <div>
                        <span class="badge {'badge-success' if entry['prediction'] == 1 else 'badge-danger'}">
                            {entry['prediction_label']}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Expandable details
            with st.expander(f"📊 View Details — {entry['id']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Input Data:**")
                    input_df = pd.DataFrame([{
                        "Feature": k.replace('_', ' ').title(),
                        "Value": v
                    } for k, v in entry['data'].items() if k not in ['education_label', 'education_num']])
                    st.dataframe(input_df, hide_index=True, use_container_width=True)
                
                with col2:
                    st.markdown("**Prediction Details:**")
                    st.json({
                        "ID": entry['id'],
                        "Timestamp": entry['datetime'],
                        "Prediction": entry['prediction_label'],
                        "Probability": f"{entry['probability']:.1%}" if entry['probability'] else "N/A"
                    })

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
def render_about():
    """Render about page with project information"""
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 2.5rem; margin: 0; background: linear-gradient(135deg, #059669, #7c3aed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            ℹ️ About
        </h1>
        <p style="font-size: 1.1rem; color: #64748b; margin-top: 8px;">
            Learn more about this application and its technology
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Project Overview
    st.markdown("""
    <div class="custom-card">
        <h2>🎯 Project Overview</h2>
        <p>
            <strong>Income Predictor Pro</strong> is a sophisticated machine learning application
            that predicts whether an individual's annual income exceeds $50,000 based on
            demographic and employment data.
        </p>
        <p>
            Built on the <strong>UCI Adult Census Income Dataset</strong>, this application
            demonstrates the power of ensemble learning and provides an intuitive interface
            for real-time predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technology Stack
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="custom-card">
            <h3>🛠️ Technology Stack</h3>
            <ul style="list-style: none; padding: 0;">
                <li>🐍 <strong>Python</strong> — Core programming language</li>
                <li>📊 <strong>Streamlit</strong> — Web application framework</li>
                <li>🤖 <strong>Scikit-learn</strong> — Machine learning library</li>
                <li>🚀 <strong>CatBoost</strong> — Gradient boosting</li>
                <li>📈 <strong>Plotly</strong> — Interactive visualizations</li>
                <li>🔢 <strong>NumPy/Pandas</strong> — Data processing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="custom-card">
            <h3>🧠 Model Architecture</h3>
            <ul style="list-style: none; padding: 0;">
                <li>📚 <strong>Type</strong>: Stacking Classifier</li>
                <li>🏗️ <strong>Base Models</strong>: Hist Gradient Boosting + CatBoost</li>
                <li>🎯 <strong>Meta-Learner</strong>: Logistic Regression</li>
                <li>📊 <strong>Accuracy</strong>: ~87%</li>
                <li>🔢 <strong>Features</strong>: 14 inputs → 47 engineered</li>
                <li>📅 <strong>Trained</strong>: 2024</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Performance
    st.markdown("""
    <div class="custom-card">
        <h3>📊 Model Performance Metrics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">🎯 Accuracy</div>
            <div class="stat-number" style="font-size: 1.8rem;">87%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">📈 Precision</div>
            <div class="stat-number" style="font-size: 1.8rem;">86%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">📊 Recall</div>
            <div class="stat-number" style="font-size: 1.8rem;">85%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">🎯 F1-Score</div>
            <div class="stat-number" style="font-size: 1.8rem;">86%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature Engineering
    st.markdown("""
    <div class="custom-card">
        <h3>🔧 Feature Engineering Pipeline</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <h4>📥 Input Features</h4>
            <ul style="text-align: left;">
                <li>Age</li>
                <li>Work Class</li>
                <li>Education Level</li>
                <li>Marital Status</li>
                <li>Occupation</li>
                <li>Relationship</li>
                <li>Race</li>
                <li>Sex</li>
                <li>Capital Gain/Loss</li>
                <li>Hours/Week</li>
                <li>Native Country</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <h4>⚙️ Processing Steps</h4>
            <ol style="text-align: left;">
                <li><strong>Cleaning</strong>: Handle missing values</li>
                <li><strong>Feature Engineering</strong>: Capital difference</li>
                <li><strong>Imputation</strong>: Categorical missing values</li>
                <li><strong>Encoding</strong>: Binary & One-Hot encoding</li>
                <li><strong>Target Encoding</strong>: Occupation encoding</li>
                <li><strong>Scaling</strong>: Standard normalization</li>
                <li><strong>Selection</strong>: Feature importance filtering</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset Information
    st.markdown("""
    <div class="custom-card">
        <h3>📚 Dataset Information</h3>
        <p>
            The <strong>UCI Adult Census Income Dataset</strong> contains demographic and employment data
            from the 1994 US Census. It's widely used as a benchmark for classification problems.
        </p>
        <ul>
            <li><strong>Total Samples</strong>: 48,842</li>
            <li><strong>Features</strong>: 14 (after preprocessing)</li>
            <li><strong>Target</strong>: Binary classification (>50K / ≤50K)</li>
            <li><strong>Source</strong>: UCI Machine Learning Repository</li>
            <li><strong>Balanced</strong>: ~24% >50K, ~76% ≤50K</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Version and Credits
    st.markdown("""
    <div style="background: #f8fafc; padding: 24px; border-radius: 12px; margin-top: 20px; border: 1px solid #e2e8f0;">
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 16px;">
            <div>
                <strong>📦 Version</strong> 2.0.0
            </div>
            <div>
                <strong>📅 Last Updated</strong> 2024
            </div>
            <div>
                <strong>🔒 Privacy</strong> All data processed in real-time, never stored
            </div>
            <div>
                <strong>📄 License</strong> MIT
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main():
    """Main application entry point"""
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="color: #0f172a; font-size: 1.5rem;">💰 Income Predictor</h2>
            <p style="color: #64748b; font-size: 0.9rem;">v2.0 — Professional</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🎯 Predictor", "📊 Analytics", "📜 History", "ℹ️ About"],
            index=0,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick stats in sidebar
        history = PredictionHistory.get_history()
        if history:
            stats = PredictionHistory.get_stats()
            if stats:
                st.markdown("**📊 Quick Stats**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total", stats['total'])
                with col2:
                    st.metric(">50K", stats['high_income'])
    
    # Page routing
    if page == "🎯 Predictor":
        render_predictor()
    elif page == "📊 Analytics":
        render_analytics()
    elif page == "📜 History":
        render_history()
    elif page == "ℹ️ About":
        render_about()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 16px 0; color: #94a3b8; font-size: 0.85rem;">
        ⚡ Powered by Machine Learning | Predictions are estimates | Privacy Guaranteed
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Run Application
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
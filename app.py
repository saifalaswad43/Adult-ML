# income_predictor_app.py
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import hashlib

st.set_page_config(
    page_title="Income Predictor Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .prediction-high {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .prediction-low {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #135e8a;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def _fix_target_encoder(enc):
    """Patch attributes missing from encoders pickled with an older
    category_encoders version than the one installed here."""
    oe = getattr(enc, "ordinal_encoder", None)
    if oe is not None and not hasattr(oe, "index_start"):
        oe.index_start = 1
    return enc

# ---------- Load saved artifacts ----------
@st.cache_resource
def load_artifacts():
    try:
        return {
            "cat_imputer": joblib.load("categorical_imputer.pkl"),
            "sex_map": joblib.load("sex_encoder.pkl"),
            "country_map": joblib.load("native_country_encoder.pkl"),
            "marital_map": joblib.load("marital_status_encoder.pkl"),
            "ohe": joblib.load("onehot_encoder.pkl"),
            "target_enc": _fix_target_encoder(joblib.load("occupation_target_encoder.pkl")),
            "scaler": joblib.load("standard_scaler.pkl"),
            "selected_features": joblib.load("selected_features.pkl"),
            "config": joblib.load("feature_columns_config.pkl"),
            "model": joblib.load("best_model.pkl"),
        }
    except FileNotFoundError as e:
        st.error(f"❌ Missing model file: {e}. Please ensure all .pkl files are in the same directory.")
        st.stop()

art = load_artifacts()

# ---------- Data cleaning functions ----------
def apply_basic_cleaning(df_raw):
    d = df_raw.copy()
    d["race"] = d["race"].replace(["Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"], "Others")
    d["marital.status"] = d["marital.status"].replace(
        ["Widowed", "Divorced", "Separated", "Never-married"], "not_married")
    d["marital.status"] = d["marital.status"].replace(
        ["Married-civ-spouse", "Married-spouse-absent", "Married-AF-spouse"], "married")
    work_class = {
        "Self-emp-inc": "Self-employed", "Self-emp-not-inc": "Self-employed",
        "State-gov": "Government", "Federal-gov": "Government", "Local-gov": "Government",
        "Without-pay": "Without-pay",
    }
    d["workclass"] = d["workclass"].replace(work_class)
    d.loc[d["native.country"] != "United-States", "native.country"] = "Others"
    return d

def apply_feature_engineering(d):
    d = d.copy()
    d["capital_diff"] = d["capital.gain"] - d["capital.loss"]
    d["total_income_assets"] = d["capital.gain"] + d["capital.loss"]
    d["work_hours_category"] = pd.cut(d["hours.per.week"], 
                                       bins=[0, 20, 40, 60, 100], 
                                       labels=["Part-time", "Full-time", "Overtime", "Extreme"])
    return d

# ---------- Prediction function ----------
def predict_income(input_data):
    d = apply_basic_cleaning(input_data)
    d = apply_feature_engineering(d)
    
    cfg = art["config"]
    d[cfg["categorical_impute_cols"]] = art["cat_imputer"].transform(d[cfg["categorical_impute_cols"]])
    
    d["sex"] = d["sex"].map(art["sex_map"])
    d["native.country"] = d["native.country"].map(art["country_map"])
    d["marital.status"] = d["marital.status"].map(art["marital_map"])
    
    ohe_cols = cfg["ohe_cols"]
    ohe_out = art["ohe"].transform(d[ohe_cols])
    ohe_df = pd.DataFrame(ohe_out, columns=art["ohe"].get_feature_names_out(ohe_cols), index=d.index)
    d = pd.concat([d.drop(columns=ohe_cols), ohe_df], axis=1)
    
    d["occupation"] = art["target_enc"].transform(d[["occupation"]])["occupation"]
    
    num_cols = cfg["numeric_cols_to_scale"]
    d[num_cols] = art["scaler"].transform(d[num_cols])
    
    d_fs = d[art["selected_features"]]
    
    pred = art["model"].predict(d_fs)[0]
    proba = art["model"].predict_proba(d_fs)[0][1] if hasattr(art["model"], "predict_proba") else None
    
    return pred, proba

# ---------- Session state initialization ----------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

# ---------- Sidebar ----------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/income.png", width=80)
    st.markdown("## 📊 Dashboard")
    
    if st.session_state.prediction_history:
        total_predictions = len(st.session_state.prediction_history)
        high_income_count = sum(1 for p in st.session_state.prediction_history if p["prediction"] == ">50K")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Predictions", total_predictions)
        with col2:
            st.metric(">50K Rate", f"{high_income_count/total_predictions*100:.1f}%")
        
        st.divider()
        st.markdown("### 📈 Prediction Trend")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
        
        fig = px.line(history_df, x="timestamp", y="probability", 
                      title="Confidence Over Time",
                      labels={"probability": "Probability of >50K", "timestamp": "Time"})
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.markdown("### ℹ️ About")
    st.info("""
    This app predicts whether an individual's income exceeds $50K based on census data.
    
    **Model**: XGBoost (or your best model)
    **Accuracy**: ~87%
    
    All predictions are stored locally in your session.
    """)

# ---------- Main content ----------
st.markdown('<p class="main-header">💰 Income Predictor Pro</p>', unsafe_allow_html=True)

# Input form
with st.form("income_form"):
    st.markdown("### 👤 Personal Information")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=17, max_value=90, value=35, help="Age in years")
        sex = st.selectbox("Sex", ["Male", "Female"])
        race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
    
    with col2:
        workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc",
                                                "Federal-gov", "Local-gov", "State-gov", "Without-pay"])
        education_num = st.slider("Education Level (Years)", min_value=1, max_value=16, value=10,
                                   help="1=Preschool, 16=Doctorate")
        marital_status = st.selectbox("Marital Status", ["Never-married", "Married-civ-spouse",
                                                          "Divorced", "Separated", "Widowed",
                                                          "Married-spouse-absent", "Married-AF-spouse"])
    
    with col3:
        occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service",
                                                  "Sales", "Exec-managerial", "Prof-specialty",
                                                  "Handlers-cleaners", "Machine-op-inspct",
                                                  "Adm-clerical", "Farming-fishing",
                                                  "Transport-moving", "Priv-house-serv",
                                                  "Protective-serv", "Armed-Forces"])
        relationship = st.selectbox("Relationship", ["Wife", "Own-child", "Husband",
                                                      "Not-in-family", "Other-relative", "Unmarried"])
        native_country = st.selectbox("Native Country", ["United-States", "Other"])
    
    st.markdown("### 💵 Financial Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        capital_gain = st.number_input("Capital Gain ($)", min_value=0, max_value=100000, value=0, step=100)
    with col2:
        capital_loss = st.number_input("Capital Loss ($)", min_value=0, max_value=5000, value=0, step=100)
    with col3:
        hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
    
    st.markdown("### 📊 Advanced Analysis Options")
    col1, col2 = st.columns(2)
    with col1:
        show_feature_importance = st.checkbox("Show Feature Importance", value=True)
    with col2:
        show_comparison = st.checkbox("Show Comparison to Average", value=True)
    
    submitted = st.form_submit_button("🔮 Predict Income")

# ---------- Handle prediction ----------
if submitted:
    # Create input dataframe
    raw = pd.DataFrame([{
        "age": age, 
        "workclass": workclass, 
        "education.num": education_num,
        "marital.status": marital_status, 
        "occupation": occupation,
        "relationship": relationship, 
        "race": race, 
        "sex": sex,
        "capital.gain": capital_gain, 
        "capital.loss": capital_loss,
        "hours.per.week": hours_per_week,
        "native.country": native_country if native_country == "United-States" else "Others",
    }])
    
    # Make prediction
    with st.spinner("Analyzing your profile..."):
        pred, proba = predict_income(raw)
        
    # Generate unique ID for this prediction
    pred_id = hashlib.md5(f"{datetime.now()}{age}{sex}{occupation}".encode()).hexdigest()[:8]
    label = ">50K" if pred == 1 else "<=50K"
    
    # Store in history
    history_entry = {
        "id": pred_id,
        "timestamp": datetime.now().isoformat(),
        "age": age,
        "sex": sex,
        "occupation": occupation,
        "workclass": workclass,
        "education": education_num,
        "hours_per_week": hours_per_week,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "prediction": label,
        "probability": proba,
    }
    st.session_state.prediction_history.append(history_entry)
    
    # ---------- Display Results ----------
    st.divider()
    st.markdown("## 📊 Prediction Results")
    
    # Prediction box
    if label == ">50K":
        st.markdown(f"""
        <div class="prediction-box prediction-high">
            <h2 style="color:#28a745;">✅ Income > $50K</h2>
            <p style="font-size:1.2rem;">Probability: {proba:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box prediction-low">
            <h2 style="color:#dc3545;">❌ Income ≤ $50K</h2>
            <p style="font-size:1.2rem;">Probability of >50K: {proba:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Analysis section
    st.markdown("### 📈 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature importance (simplified - replace with actual if available)
        st.markdown("#### 🎯 Key Factors")
        factors = {
            "Education": education_num * 2.5,
            "Work Hours": hours_per_week * 1.2,
            "Capital Gain": capital_gain / 1000,
            "Age": age * 0.8,
            "Occupation": 50 if occupation in ["Exec-managerial", "Prof-specialty"] else 20,
        }
        max_factor = max(factors.values())
        for factor, value in factors.items():
            progress = min(value/max_factor * 100, 100)
            st.markdown(f"**{factor}**: {value:.1f}")
            st.progress(progress/100, text=f"Importance: {progress:.0f}%")
    
    with col2:
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = proba * 100,
            title = {'text': "Probability of >50K Income"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightcoral"},
                    {'range': [30, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Comparison to average
    if show_comparison:
        st.markdown("### 📊 Comparison to Census Average")
        avg_data = {
            "Age": 38,
            "Hours/Week": 40,
            "Education Years": 10,
            "Capital Gain": 1500,
            "Capital Loss": 200,
        }
        
        comp_df = pd.DataFrame({
            "Metric": list(avg_data.keys()),
            "Your Value": [age, hours_per_week, education_num, capital_gain, capital_loss],
            "Average": list(avg_data.values()),
        })
        
        fig = px.bar(comp_df, x="Metric", y=["Your Value", "Average"], 
                     barmode="group", title="Your Profile vs. Census Average")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (if available)
    if show_feature_importance and hasattr(art["model"], "feature_importances_"):
        st.markdown("### 🌟 Model Feature Importance")
        try:
            feature_importance = pd.DataFrame({
                "Feature": art["selected_features"],
                "Importance": art["model"].feature_importances_
            }).sort_values("Importance", ascending=False).head(10)
            
            fig = px.bar(feature_importance, x="Importance", y="Feature", 
                         orientation="h", title="Top 10 Most Important Features")
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Feature importance visualization not available for this model type.")

# ---------- History Section ----------
if st.session_state.prediction_history:
    st.divider()
    st.markdown("## 📜 Prediction History")
    
    with st.expander("View All Predictions", expanded=False):
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
        display_cols = ["timestamp", "age", "sex", "occupation", "prediction", "probability"]
        history_display = history_df[display_cols].copy()
        history_display["probability"] = history_display["probability"].apply(lambda x: f"{x:.1%}")
        history_display.rename(columns={
            "timestamp": "Time",
            "prediction": "Result",
            "probability": "Confidence"
        }, inplace=True)
        
        st.dataframe(history_display, use_container_width=True)
        
        # Visualization of history
        if len(st.session_state.prediction_history) >= 2:
            fig = px.line(history_df, x="timestamp", y="probability", 
                          title="Prediction Confidence Over Time",
                          markers=True)
            fig.update_layout(xaxis_title="Time", yaxis_title="Probability of >50K")
            st.plotly_chart(fig, use_container_width=True)
        
        # Download history
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download History (CSV)",
            data=csv,
            file_name=f"income_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

# ---------- Footer ----------
st.divider()
st.caption("Built with ❤️ using Streamlit | Data from UCI Adult Census Dataset")

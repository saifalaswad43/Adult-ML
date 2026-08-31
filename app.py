import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Income Predictor", page_icon="💰")
st.title("💰 Adult Income Predictor (>50K or <=50K)")

# ---------- Load saved artifacts (from the notebook, Section 9-13) ----------
@st.cache_resource
def load_artifacts():
    return {
        "cat_imputer": joblib.load("categorical_imputer.pkl"),
        "sex_map": joblib.load("sex_encoder.pkl"),
        "country_map": joblib.load("native_country_encoder.pkl"),
        "marital_map": joblib.load("marital_status_encoder.pkl"),
        "ohe": joblib.load("onehot_encoder.pkl"),
        "target_enc": joblib.load("occupation_target_encoder.pkl"),
        "scaler": joblib.load("standard_scaler.pkl"),
        "selected_features": joblib.load("selected_features.pkl"),
        "config": joblib.load("feature_columns_config.pkl"),
        "model": joblib.load("best_model.pkl"),
    }

art = load_artifacts()

# ---------- Same cleaning / feature-engineering functions as the notebook ----------
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
    return d

# ---------- Input form ----------
with st.form("income_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 17, 90, 35)
        workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc",
                                                "Federal-gov", "Local-gov", "State-gov", "Without-pay"])
        education_num = st.slider("Education (num)", 1, 16, 10)
        marital_status = st.selectbox("Marital status", ["Never-married", "Married-civ-spouse",
                                                          "Divorced", "Separated", "Widowed",
                                                          "Married-spouse-absent", "Married-AF-spouse"])
        occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service",
                                                  "Sales", "Exec-managerial", "Prof-specialty",
                                                  "Handlers-cleaners", "Machine-op-inspct",
                                                  "Adm-clerical", "Farming-fishing",
                                                  "Transport-moving", "Priv-house-serv",
                                                  "Protective-serv", "Armed-Forces"])
        relationship = st.selectbox("Relationship", ["Wife", "Own-child", "Husband",
                                                      "Not-in-family", "Other-relative", "Unmarried"])
    with col2:
        race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander",
                                      "Amer-Indian-Eskimo", "Other"])
        sex = st.selectbox("Sex", ["Male", "Female"])
        capital_gain = st.number_input("Capital gain", 0, 100000, 0)
        capital_loss = st.number_input("Capital loss", 0, 5000, 0)
        hours_per_week = st.number_input("Hours per week", 1, 99, 40)
        native_country = st.selectbox("Native country", ["United-States", "Other"])

    submitted = st.form_submit_button("Predict")

if submitted:
    raw = pd.DataFrame([{
        "age": age, "workclass": workclass, "education.num": education_num,
        "marital.status": marital_status, "occupation": occupation,
        "relationship": relationship, "race": race, "sex": sex,
        "capital.gain": capital_gain, "capital.loss": capital_loss,
        "hours.per.week": hours_per_week,
        "native.country": native_country if native_country == "United-States" else "Others",
    }])

    d = apply_basic_cleaning(raw)
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

    d["occupation"] = art["target_enc"].transform(d["occupation"])

    num_cols = cfg["numeric_cols_to_scale"]
    d[num_cols] = art["scaler"].transform(d[num_cols])

    d_fs = d[art["selected_features"]]

    pred = art["model"].predict(d_fs)[0]
    label = ">50K" if pred == 1 else "<=50K"

    st.subheader(f"Predicted income: **{label}**")
    if hasattr(art["model"], "predict_proba"):
        proba = art["model"].predict_proba(d_fs)[0][1]
        st.write(f"Probability of >50K: {proba:.2%}")

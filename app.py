"""
Streamlit app — Adult Income Prediction (>50K / <=50K)
--------------------------------------------------------
Reconstructs the exact deployment pipeline built in the training notebook:

    raw form input
        -> apply_basic_cleaning()
        -> apply_feature_engineering()   (adds capital_diff)
        -> categorical_imputer           (occupation, workclass, native.country)
        -> sex_encoder / native_country_encoder / marital_status_encoder (maps)
        -> onehot_encoder                (workclass, relationship, race)
        -> occupation_target_encoder
        -> standard_scaler
        -> selected_features subset
        -> best_model.predict / predict_proba

All artifacts are the exact ones saved with joblib during training, so no
re-fitting happens here — the app only replays the saved transformations.
"""

import glob
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Adult Income Predictor", page_icon="💰", layout="centered")

ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# Helpers copied 1:1 from the training notebook so the raw-row transform is
# identical to what the saved artifacts were fit on.
# --------------------------------------------------------------------------
def apply_basic_cleaning(df_raw: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    d = df_raw.copy()
    d.replace("?", np.nan, inplace=True)

    if is_training:
        d.drop_duplicates(keep="first", inplace=True)
        d.reset_index(drop=True, inplace=True)

    for col in ["fnlwgt", "education"]:
        if col in d.columns:
            d.drop(columns=[col], inplace=True)

    if is_training:
        d.drop_duplicates(keep="first", inplace=True)
        d.reset_index(drop=True, inplace=True)

    if "race" in d.columns:
        d["race"] = d["race"].replace(
            ["Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"], "Others"
        )

    if "marital.status" in d.columns:
        d["marital.status"] = d["marital.status"].replace(
            ["Widowed", "Divorced", "Separated", "Never-married"], "not_married"
        )
        d["marital.status"] = d["marital.status"].replace(
            ["Married-civ-spouse", "Married-spouse-absent", "Married-AF-spouse"], "married"
        )

    if "workclass" in d.columns:
        work_class = {
            "Self-emp-inc": "Self-employed",
            "Self-emp-not-inc": "Self-employed",
            "State-gov": "Government",
            "Federal-gov": "Government",
            "Local-gov": "Government",
            "Without-pay": "Without-pay",
        }
        d["workclass"] = d["workclass"].replace(work_class)

    if "native.country" in d.columns:
        d.loc[
            d["native.country"].notna() & (d["native.country"] != "United-States"),
            "native.country",
        ] = "Others"

    return d


def apply_feature_engineering(df_clean: pd.DataFrame) -> pd.DataFrame:
    d = df_clean.copy()
    d["capital_diff"] = d["capital.gain"] - d["capital.loss"]
    return d


# --------------------------------------------------------------------------
# Load every saved artifact (cached so it only happens once per session).
# Files are located by suffix so the original long uploaded filenames work
# as-is, without needing to be renamed.
# --------------------------------------------------------------------------
def _find(suffix: str) -> str:
    matches = sorted(glob.glob(os.path.join(ARTIFACT_DIR, f"*{suffix}")))
    if not matches:
        raise FileNotFoundError(
            f"Could not find a file ending in '{suffix}' next to app.py"
        )
    return matches[0]


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    return {
        "categorical_imputer": joblib.load(_find("categorical_imputer.pkl")),
        "sex_encoder": joblib.load(_find("sex_encoder.pkl")),
        "native_country_encoder": joblib.load(_find("native_country_encoder.pkl")),
        "marital_status_encoder": joblib.load(_find("marital_status_encoder.pkl")),
        "onehot_encoder": joblib.load(_find("onehot_encoder.pkl")),
        "occupation_target_encoder": joblib.load(_find("occupation_target_encoder.pkl")),
        "scaler": joblib.load(_find("standard_scaler.pkl")),
        "selected_features": joblib.load(_find("selected_features.pkl")),
        "feature_columns_config": joblib.load(_find("feature_columns_config.pkl")),
        "model": joblib.load(_find("best_model.pkl")),
    }


def predict_income(raw_row: pd.DataFrame, artifacts: dict):
    cfg = artifacts["feature_columns_config"]

    sim = apply_basic_cleaning(raw_row, is_training=False)
    sim = apply_feature_engineering(sim)

    # Categorical imputation (occupation / workclass / native.country)
    sim[cfg["categorical_impute_cols"]] = artifacts["categorical_imputer"].transform(
        sim[cfg["categorical_impute_cols"]]
    )

    # Binary maps
    sim["sex"] = sim["sex"].map(artifacts["sex_encoder"])
    sim["native.country"] = sim["native.country"].map(artifacts["native_country_encoder"])
    sim["marital.status"] = sim["marital.status"].map(artifacts["marital_status_encoder"])

    # One-hot encoding
    ohe_cols = cfg["ohe_cols"]
    ohe = artifacts["onehot_encoder"]
    encoded = ohe.transform(sim[ohe_cols])
    encoded = pd.DataFrame(
        encoded, columns=ohe.get_feature_names_out(ohe_cols), index=sim.index
    )
    sim = pd.concat([sim.drop(columns=ohe_cols), encoded], axis=1)

    # Target encoding for occupation
    sim["occupation"] = artifacts["occupation_target_encoder"].transform(sim["occupation"])

    # Scaling
    num_cols = cfg["numeric_cols_to_scale"]
    sim[num_cols] = artifacts["scaler"].transform(sim[num_cols])

    # Final feature subset, in the exact order the model expects
    sim_fs = sim[artifacts["selected_features"]]

    pred = artifacts["model"].predict(sim_fs)[0]
    proba = None
    if hasattr(artifacts["model"], "predict_proba"):
        proba = artifacts["model"].predict_proba(sim_fs)[0]

    return pred, proba


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("💰 Adult Income Predictor")
st.caption(
    "Predicts whether a person's annual income is **above** or **at/below $50K**, "
    "based on the UCI Adult / Census Income dataset pipeline."
)

try:
    artifacts = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"{e}\n\nMake sure all the `.pkl` artifact files are in the same folder as `app.py`."
    )
    st.stop()

EDUCATION_MAP = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5,
    "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10,
    "Assoc-voc": 11, "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14,
    "Prof-school": 15, "Doctorate": 16,
}

# These are already the FINAL, consolidated categories the trained pipeline
# uses internally (see apply_basic_cleaning in the notebook: workclass and
# race are collapsed into these exact buckets). Showing them directly here
# keeps the form short and skips a redundant extra mapping step.
WORKCLASS_OPTIONS = ["Private", "Self-employed", "Government", "Without-pay", "Never-worked"]
MARITAL_OPTIONS = ["married", "not_married"]
OCCUPATION_OPTIONS = [
    "Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial",
    "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct",
    "Other-service", "Priv-house-serv", "Prof-specialty", "Protective-serv",
    "Sales", "Tech-support", "Transport-moving",
]
RELATIONSHIP_OPTIONS = [
    "Husband", "Wife", "Own-child", "Not-in-family", "Other-relative", "Unmarried",
]
RACE_OPTIONS = ["White", "Black", "Others"]
SEX_OPTIONS = ["Male", "Female"]
NATIVE_COUNTRY_OPTIONS = ["United-States", "Other"]

with st.form("income_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 17, 90, 35)
        workclass = st.selectbox("Workclass", WORKCLASS_OPTIONS)
        education = st.selectbox(
            "Education", list(EDUCATION_MAP.keys()), index=list(EDUCATION_MAP.keys()).index("Bachelors")
        )
        marital_status = st.selectbox("Marital status", MARITAL_OPTIONS)
        occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS)
        relationship = st.selectbox("Relationship", RELATIONSHIP_OPTIONS)

    with col2:
        race = st.selectbox("Race", RACE_OPTIONS)
        sex = st.selectbox("Sex", SEX_OPTIONS)
        capital_gain = st.number_input("Capital gain", min_value=0, max_value=99999, value=0, step=100)
        capital_loss = st.number_input("Capital loss", min_value=0, max_value=4356, value=0, step=50)
        hours_per_week = st.slider("Hours per week", 1, 99, 40)
        native_country = st.selectbox("Native country", NATIVE_COUNTRY_OPTIONS)

    submitted = st.form_submit_button("Predict income", use_container_width=True)

if submitted:
    raw_row = pd.DataFrame([{
        "age": age,
        "workclass": workclass,
        "education.num": EDUCATION_MAP[education],
        "marital.status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital.gain": capital_gain,
        "capital.loss": capital_loss,
        "hours.per.week": hours_per_week,
        "native.country": native_country,
    }])

    pred, proba = predict_income(raw_row, artifacts)
    label = ">50K" if pred == 1 else "<=50K"

    st.divider()
    if pred == 1:
        st.success(f"### Predicted income: **{label}** 🎉")
    else:
        st.info(f"### Predicted income: **{label}**")

    if proba is not None:
        st.write("Prediction confidence:")
        st.progress(float(proba[1]), text=f">50K probability: {proba[1]:.1%}")
        st.caption(f"<=50K probability: {proba[0]:.1%}")

    with st.expander("Show processed input row sent to the model"):
        st.dataframe(raw_row)

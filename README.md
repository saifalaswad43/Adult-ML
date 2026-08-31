<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=220&section=header&text=Income%20Predictor%20Pro&fontSize=48&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Predict%20Income%20%3E%2450K%20with%20Machine%20Learning&descAlignY=58&descSize=18" width="100%"/>

<a href="https://github.com/saifalaswad43/Adult-ML">
  <img src="https://readme-typing-svg.demolab.com/?lines=%F0%9F%92%B0+AI-Powered+Income+Classifier;%F0%9F%A4%96+Trained+on+the+UCI+Adult+Census+Dataset;%E2%9A%A1+XGBoost+%7C+Scikit-learn+%7C+Streamlit;%F0%9F%93%8A+~87%25+Accuracy+on+Test+Data&font=Fira+Code&center=true&width=700&height=50&color=764ABA&vCenter=true&size=22&pause=1500"/>
</a>

<br/>

<img src="https://img.shields.io/github/stars/saifalaswad43/Adult-ML?style=for-the-badge&color=FFD700&logo=github" />
<img src="https://img.shields.io/github/forks/saifalaswad43/Adult-ML?style=for-the-badge&color=8A2BE2&logo=github" />
<img src="https://img.shields.io/github/last-commit/saifalaswad43/Adult-ML?style=for-the-badge&color=00C7B7&logo=git" />
<img src="https://img.shields.io/github/languages/top/saifalaswad43/Adult-ML?style=for-the-badge&color=blue&logo=python" />
<img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" />

<br/><br/>

<img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/XGBoost-EC4E20?style=flat-square&logo=xgboost&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/LightGBM-02569B?style=flat-square" />
<img src="https://img.shields.io/badge/CatBoost-FFCC00?style=flat-square&logoColor=black" />
<img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" />

</div>

<br/>

## 📌 Overview

**Income Predictor Pro** is a machine learning web app that predicts whether a person's annual income exceeds **$50K**, based on demographic and employment attributes from the classic **UCI Adult Census Income Dataset**. It ships with a full **Streamlit** interface — form-based prediction, live probability gauges, historical tracking, and interactive analytics — backed by a tuned **XGBoost** classifier.

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/216122041-518ac897-8d92-4c6b-9b3f-ca01dcaf38ee.gif" width="500">
</div>

---

## ✨ Features

<table>
<tr>
<td width="50%" valign="top">

### 🏠 Home — Prediction
- Full personal & financial input form
- Real-time inference with confidence score
- Animated result cards (high / low income)
- Quick contributing-factor breakdown

</td>
<td width="50%" valign="top">

### 📊 Analysis
- Live probability gauge (Plotly)
- Feature contribution bars
- Comparison vs. census averages
- Occupation income-tier grouping

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 📜 History
- Session-based prediction log
- Filter / sort by confidence, date, age
- Trend charts & prediction distribution
- One-click CSV export

</td>
<td width="50%" valign="top">

### ℹ️ About
- Dataset & model documentation
- Performance metrics at a glance
- Tech stack badges
- FAQ section

</td>
</tr>
</table>

---

## 🧠 Model Pipeline

```mermaid
flowchart LR
    A[Raw Adult Census Data] --> B[Data Cleaning]
    B --> C[Feature Engineering]
    C --> D[Encoding<br/>OneHot · Target · Label Maps]
    D --> E[Standard Scaling]
    E --> F[Feature Selection]
    F --> G[XGBoost Classifier]
    G --> H[Prediction + Probability]
```

**Engineered features:** `capital_diff`, `capital_ratio`, `total_capital`, `education_hours`, `age_education`

| Metric | Score |
|:--|:--:|
| 🎯 Accuracy | **~87%** |
| 🎯 Precision | **~85%** |
| 🎯 Recall | **~82%** |
| 🎯 F1-Score | **~83%** |

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/saifalaswad43/Adult-ML.git
cd Adult-ML
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501` 🎉

---

## 📂 Project Structure

```
Adult-ML/
├── app.py                          # Streamlit application
├── Adult.ipynb                     # EDA, training & experimentation notebook
├── adult.csv                       # UCI Adult Census dataset
├── best_model.pkl                  # Trained XGBoost model
├── categorical_imputer.pkl         # Missing-value imputer
├── onehot_encoder.pkl              # One-hot encoder
├── occupation_target_encoder.pkl   # Target encoder (occupation)
├── marital_status_encoder.pkl      # Marital status map
├── native_country_encoder.pkl      # Native country map
├── sex_encoder.pkl                 # Sex map
├── standard_scaler.pkl             # Feature scaler
├── selected_features.pkl           # Final feature set
├── feature_columns_config.pkl      # Column configuration
└── requirements.txt                # Dependencies
```

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://skillicons.dev/icons?i=python)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/-XGBoost-EC4E20?style=for-the-badge)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

</div>

---

## 📊 Dataset

**Source:** [UCI Machine Learning Repository — Adult Census Income](https://archive.ics.uci.edu/dataset/2/adult)
**Size:** 48,842 records · 14 attributes
**Target:** Binary classification — income `>50K` or `<=50K`

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer" width="100%"/>

</div>

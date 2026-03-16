# 🌍 Predicting Economic Freedom Index Classifications Using Machine Learning
### Multi-Class Classification, Ensemble Modeling, and ANN on a Decade of Country-Level Economic Data

> A supervised machine learning pipeline that predicts a country's economic freedom classification tier from institutional and policy indicators, benchmarking 10 models including tree ensembles, boosting methods, and a custom-built Artificial Neural Network across 10 years of Heritage Foundation data.

---

## 📌 Overview

Economic freedom — the degree to which individuals and businesses can make their own economic decisions — is one of the most closely watched macro-level indicators among policymakers, investors, and researchers. The Index of Economic Freedom, published annually by The Heritage Foundation, ranks countries across a wide range of economic and institutional dimensions.

This project frames the prediction of a country's economic freedom classification tier (Free, Mostly Free, Moderately Free, Mostly Unfree, and Repressed) as a supervised multi-class classification problem. Using a decade of country-level data spanning 2013 to 2022, we trained and evaluated 10 classification models covering traditional ML algorithms and a feedforward Artificial Neural Network, to understand which features best predict a country's economic standing and how accurately each approach can classify it.

| Goal | Approach |
|------|----------|
| Predict economic freedom classification tier from country indicators | Multi-class classification across 10 supervised ML models |
| Identify which economic and institutional features drive classification | Feature importance analysis via Random Forest and XGBoost |
| Evaluate model robustness across different algorithm families | Benchmarking from Logistic Regression to Gradient Boosting to ANN |
| Surface insights through an interactive interface | Streamlit web app for real-time country-level predictions |

---

## 📂 Dataset

**Index of Economic Freedom (2013–2022)**
Source: [The Heritage Foundation](https://www.heritage.org/index/)

- **Coverage:** 10 years of country-level data across global economies
- **Target variable:** Economic freedom classification tier (5 classes: Free, Mostly Free, Moderately Free, Mostly Unfree, Repressed)

### Feature Dimensions

| Dimension | Example Features |
|-----------|-----------------|
| **Rule of Law** | `Property Rights`, `Judicial Effectiveness`, `Government Integrity` |
| **Government Size** | `Tax Burden`, `Government Spending`, `Fiscal Health` |
| **Regulatory Efficiency** | `Business Freedom`, `Labour Freedom`, `Monetary Freedom` |
| **Market Openness** | `Trade Freedom`, `Investment Freedom`, `Financial Freedom` |
| **Target** | `Overall Economic Freedom Score`, `Classification Tier` |

### Dataset Files

| File | Purpose |
|------|---------|
| `Classification_Dataset.xlsx` | Raw, unprocessed source data |
| `Visualisation_Dataset.xlsx` | Cleaned subset for EDA and visualisations |
| `preprocessed_data.csv` | Feature-engineered, encoded, and scaled data for model training |
| `TimeSeries_Dataset.xlsx` | Temporal version for time-aware analysis |

---

## 🔧 Tech Stack

| Category | Libraries / Tools |
|----------|-------------------|
| Data Manipulation | `pandas`, `numpy` |
| Machine Learning | `scikit-learn`, `xgboost` |
| Deep Learning | `TensorFlow`, `Keras` |
| Visualisation | `matplotlib`, `seaborn` |
| Web App | `Streamlit` |
| Notebook Environment | `Jupyter Notebook` |
| Data Format | `CSV`, `XLSX` |

---

## 🗂️ Repository Structure

```
📦 economic-freedom-index-prediction
 ┣ 📜 Preprocessing_code.ipynb       # Data cleaning, encoding, feature engineering
 ┣ 📜 training_and_testing.ipynb     # Train and evaluate all ML models (except ANN)
 ┣ 📜 ANN.ipynb                      # Feedforward ANN training and evaluation
 ┣ 📜 XGBOOST_predictor.ipynb        # Dedicated XGBoost pipeline
 ┣ 📜 app.py                         # Streamlit dashboard for real-time predictions
 ┣ 📜 test_Results.py                # Full model score comparisons
 ┣ 📂 data
 ┃ ┣ 📜 Classification_Dataset.xlsx
 ┃ ┣ 📜 Visualisation_Dataset.xlsx
 ┃ ┣ 📜 preprocessed_data.csv
 ┃ ┗ 📜 TimeSeries_Dataset.xlsx
 ┣ 📜 model_scores.csv               # Compiled evaluation metrics across all models
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
```

---

## 🔬 Methodology

### 1. Data Preprocessing
- Missing values handled via mean/median imputation and forward-fill for time-series continuity
- Categorical target labels (economic freedom tiers) encoded using ordinal encoding
- Feature scaling applied using `StandardScaler` and `MinMaxScaler` for distance-sensitive and gradient-based models
- Train-test split performed with stratification to preserve class distributions across splits

### 2. Exploratory Data Analysis
- Correlation heatmaps across all 12 economic freedom indicators and the target score
- Distribution analysis of classification tiers across years and geographic regions
- Feature importance pre-screening using Random Forest to guide model selection

### 3. Model Benchmarking — 10 Classification Models

| # | Model |
|---|-------|
| 1 | Logistic Regression |
| 2 | K-Nearest Neighbours (KNN) |
| 3 | Decision Tree Classifier |
| 4 | Random Forest Classifier |
| 5 | Gradient Boosting Classifier |
| 6 | XGBoost Classifier |
| 7 | Support Vector Machine (SVM) |
| 8 | Naïve Bayes |
| 9 | Extra Trees Classifier |
| 10 | Artificial Neural Network (ANN) |

All models evaluated on Accuracy, Precision, Recall, F1-Score (macro and weighted), Confusion Matrix analysis, and ROC-AUC (one-vs-rest for multi-class).

### 4. Deep Learning — Artificial Neural Network (ANN)

A feedforward ANN was implemented separately in `ANN.ipynb` using Keras/TensorFlow:

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

> **Key Insight:** Dropout regularisation and early stopping were critical for preventing overfitting on the moderate-sized dataset. The Softmax output layer handles the 5-class classification directly, producing calibrated class probabilities per country.

### 5. Evaluation

All models were assessed on the held-out test set across the following metrics:

| Metric | What It Measures |
|--------|-----------------|
| **Accuracy** | Overall correct classifications |
| **Precision / Recall / F1** | Per-class and macro-averaged performance |
| **Confusion Matrix** | Misclassification patterns across tiers |
| **ROC-AUC** | One-vs-rest discrimination ability across all 5 classes |

---

## 📊 Key Results

- Ensemble and boosting models (Random Forest, XGBoost, Gradient Boosting) consistently outperformed linear and distance-based baselines across all evaluation metrics
- Tree-based ensembles demonstrated strong generalisation on held-out test data, with XGBoost achieving top performance among individual models
- The ANN achieved competitive performance, particularly on the larger feature-set configurations, with Dropout and early stopping keeping overfitting in check
- Rule of Law and Regulatory Efficiency indicators emerged as the strongest predictors of economic freedom classification
- Full model score comparisons are available in `model_scores.csv` and `test_Results.py`

---

## ⚠️ Limitations

- The dataset contains moderate class imbalance across economic freedom tiers, which may bias predictions toward majority classes
- Temporal dependencies between years are partially addressed but a full time-series forecasting approach could better capture year-over-year trends
- The model is trained on country-level aggregates; sub-national or regional granularity is not captured
- Some features from The Heritage Foundation involve qualitative expert assessments, introducing an element of subjectivity into the feature space

---

## 🔮 Future Work

- **LSTM and Transformer-based models** — leverage the full temporal structure of the 10-year dataset for sequence-aware predictions
- **SHAP values** — apply SHapley Additive exPlanations for deeper model interpretability and per-country feature attribution
- **Semi-supervised learning** — explore transfer learning approaches to handle data-scarce years and newer countries
- **Real-time dashboard** — extend the Streamlit app with live predictions on newly released Heritage Foundation annual data
- **Sub-national analysis** — incorporate regional or city-level economic indicators where available to increase prediction granularity

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/d4h2nu8h/NUS_Economic_Freedom_Index_Prediction.git
cd NUS_Economic_Freedom_Index_Prediction

# Install dependencies
pip install -r requirements.txt
```

Run the notebooks in the following order:

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `Preprocessing_code.ipynb` | Data cleaning, encoding, and feature engineering |
| 2 | `training_and_testing.ipynb` | Train and evaluate all ML models except ANN |
| 3 | `ANN.ipynb` | Train and evaluate the Artificial Neural Network |
| 4 | `XGBOOST_predictor.ipynb` | Dedicated XGBoost training and prediction pipeline |

To launch the Streamlit app:

```bash
streamlit run app.py
```

---

## 👤 Author

Madhumitha Rajagopal

---

## 📄 License

This project is for educational and research purposes.

# Cardio Risk Prediction Suite

## 1. Project Description

This project provides a comprehensive machine learning and deep learning solution for cardiac risk prediction using clinical and physiological data. The system is designed to assist healthcare professionals in identifying patients at risk of critical cardiac events by leveraging multiple models (XGBoost, CNN, LSTM) and an ensemble approach.

- **Purpose:**  
  To automate and enhance the accuracy of cardiac risk stratification using advanced ML/DL models, supporting clinical decision-making and early intervention.

- **Problem Solved:**  
  Manual risk assessment is time-consuming and prone to error. This project provides automated, data-driven predictions for patient risk levels (Stable, Moderate Risk, Critical Risk), improving efficiency and reliability.

- **Datasets & Preprocessing:**  
  - Clinical data (demographics, lab results, ECG, etc.) and physiological time-series (ECG signals, vital signs).
  - Preprocessing includes feature engineering (ratios, derived features), normalization, categorical encoding, and advanced balancing (SMOTE, ADASYN, manual balancing).
  - See notebooks for detailed EDA and preprocessing steps.

---

## 2. Notebook Overview

The `notebooks/` directory contains the main development and analysis notebooks:

- **Cardio_XGBoost.ipynb:**  
  - Data loading, cleaning, and feature engineering for clinical data.
  - Exploratory Data Analysis (EDA) with visualizations.
  - Model training (XGBoost, Random Forest, LightGBM), hyperparameter tuning, and evaluation.
  - Ensemble creation and performance comparison.
  - Visualizations: confusion matrices, SHAP feature importance, class distributions.

- **Cardio_CNN.ipynb:**  
  - ECG signal extraction and preprocessing.
  - Clustering and risk labeling of ECG signals.
  - CNN model training for risk classification from raw ECG.
  - Data balancing, augmentation, and evaluation.
  - Visualizations: sample ECG plots, cluster analysis, training curves.

- **2 label Cardio_LSTM_1.ipynb:**  
  - Time-series data preparation (vital signs).
  - LSTM model architecture and training for sequence classification.
  - Data balancing, augmentation, and evaluation.
  - Visualizations: training/validation curves, confusion matrices.

---

## 3. Model Details

### XGBoost (Ensemble)
- **Input:** Clinical features (demographics, labs, ECG, etc.).
- **Architecture:**  
  - XGBoost classifier with hyperparameters:  
    `n_estimators=300`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=0.1`.
  - Combined with Random Forest and LightGBM in a voting ensemble.
- **Output:** Risk class (Stable, Moderate, Critical).
- **Performance:** High accuracy and F1-score (see Evaluation Summary).

### CNN (EnhancedECG_CNN)
- **Input:** 1D ECG signal (length 5000).
- **Architecture:**  
  - 4 convolutional blocks (Conv1D, BatchNorm, MaxPool, Dropout).
  - Global average and max pooling.
  - Fully connected layers with dropout and batch normalization.
  - Output: 3-class softmax (Critical Risk, Stable, Moderate Risk).
- **Hyperparameters:** Dropout=0.3–0.4, batch size=64, epochs=80.
- **Performance:** Best validation F1: 0.8387 (see `models/CNN/model_info.txt`).

### LSTM (ImprovedLSTMClassifier)
- **Input:** Time-series of 6 vital signs (sequence length 60).
- **Architecture:**  
  - 2-layer bidirectional LSTM (hidden size=64, dropout=0.3).
  - Multihead attention (4 heads).
  - Fully connected classifier with LayerNorm and ReLU.
- **Output:** Binary or multi-class risk prediction.
- **Performance:** See Evaluation Summary.

---

## 4. Ensemble and Inference Code

- **Model Loading:**  
  - Models and preprocessors are loaded from the `models/` directory using `joblib` and `torch`.
  - The XGBoost ensemble uses a pre-trained voting classifier and a preprocessor for feature transformation.

- **API Endpoints:**  
  Implemented in `main.py` using FastAPI:
  - `POST /XGBOOST`: Predicts risk from clinical features.
  - `POST /CNN`: Predicts risk from ECG signal.
  - `POST /LSTM`: Predicts risk from vital sign time-series.

- **Prediction Logic:**  
  - Input is validated and preprocessed.
  - Each model returns a risk class label.
  - The ensemble combines model outputs (majority vote or probability averaging).

- **Example Output:**
  ```json
  {
    "predicted_label": "Critical Risk"
  }
  ```

---

## 5. Evaluation Summary

- **Metrics:**  
  - Accuracy, Precision, Recall, F1-score (macro/weighted), Confusion Matrix.
  - SHAP and feature importance for interpretability.

- **CNN (ECG):**  
  - Best Validation F1: 0.8387
  - Balanced class distribution after manual balancing.

- **XGBoost Ensemble:**  
  - Accuracy: ~0.99 (on balanced data)
  - Macro F1: ~0.99
  - See confusion matrices and classification reports in the notebooks.

- **LSTM:**  
  - High recall for critical risk class.
  - See notebook for detailed metrics and confusion matrix.

---

## 6. Running the Endpoint Locally

### Prerequisites
- Python 3.12 recommended
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```

### Running the API
- Use the provided batch script:
  ```
  setup_and_run.bat
  ```
  Or manually:
  ```
  uvicorn main:app --host 0.0.0.0 --port 8000
  ```

### Example Request (XGBoost Endpoint)
```bash
curl -X POST "http://localhost:8000/XGBOOST" -H "Content-Type: application/json" -d '{
  "age": 65,
  "sex": "M",
  "bmi": 28.5,
  "age_group": "senior",
  "bmi_category": "overweight",
  "preop_htn": "Y",
  "preop_dm": "N",
  "preop_ecg": "Normal Sinus Rhythm",
  "preop_pft": "Normal",
  "preop_hb": 12.5,
  "preop_plt": 250000,
  "preop_pt": 12.8,
  "preop_aptt": 28.5,
  "preop_na": 140,
  "preop_k": 4.2,
  "preop_gluc": 110,
  "preop_alb": 3.8,
  "preop_ast": 28,
  "preop_alt": 32,
  "preop_bun": 18,
  "preop_cr": 1.1
}'
```

### Example Response
```json
"Critical Risk"
```

---

## 7. Directory Structure

```
cardio/
  main.py
  requirements.txt
  setup_and_run.bat
  test.py
  models/
    CNN/
      enhanced_ecg_cnn_model.pth
      model_info.txt
      preprocessing_config.pkl
      scaler.pkl
    LSTM/
      model.pth
      scaler.pkl
    XGBOOST/
      cardiac_risk_ensemble_model.pkl
      cardiac_risk_preprocessor.pkl
      cardiac_risk_smote.pkl
      pure_xgboost_model.pkl
  notebooks/
    Cardio_XGBoost.ipynb
    Cardio_CNN.ipynb
    2 label Cardio_LSTM_1.ipynb
```

---

## 8. Author and Credits

- **Author:** Abdallah Abouomar (and contributors)
- **References:**  
  - XGBoost, PyTorch, scikit-learn, FastAPI documentation.
  - Public cardiac datasets and open-source ECG/vital sign resources.
  - Special thanks to the open-source community for tools and inspiration.

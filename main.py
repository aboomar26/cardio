import os
import io
import pickle
import torch
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from typing import List, Dict
import torch.nn.functional as F


app = FastAPI()

model_dir = r"D:\Projects\Cardiac Patient Monitoring System\models"

############################Xgboost###############

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


loaded_model = joblib.load(os.path.join(model_dir, r'XGBOOST\cardiac_risk_ensemble_model.pkl'))
loaded_preprocessor = joblib.load(os.path.join(model_dir, r'XGBOOST\cardiac_risk_preprocessor.pkl'))

risk_levels = {0: 'Stable', 1: 'Moderate Risk', 2: 'Critical Risk'}

# --- Categories ---
sex_categories = ['F', 'M']
preop_htn_categories = ['N', 'Y']
preop_dm_categories = ['N', 'Y']
preop_ecg_categories = [
    "Normal Sinus Rhythm",
    "Left anterior fascicular block",
    "1st degree A-V block, Left bundle branch block",
    "1st degree A-V block",
    "Atrial fibrillation",
    "Incomplete right bundle branch block, Left anterior fascicular block",
    "Atrial fibrillation, Right bundle branch block",
    "Premature supraventricular and ventricular complexes, Right bundle branch block",
    "Atrial fibrillation with slow ventricular response",
    "Right bundle branch block",
    "Incomplete right bundle branch block",
    "Left anterior hemiblock",
    "Atrial fibrillation with rapid ventricular response",
    "Premature ventricular complexes",
    "Left posterior fascicular block",
    "Atrial fibrillation with premature ventricular, Incomplete left bundle block",
    "Premature atrial complexes",
    "1st degree A-V block with Premature supraventricular complexes, Left bundle branch block",
    "1st degree A-V block with Premature atrial complexes",
    "Atrial fibrillation with premature ventricular or aberrantly conducted complexes",
    "Atrial flutter with 2:1 A-V conduction",
    "Premature supraventricular complexes",
    "Electronic ventricular pacemaker",
    "AV sequential or dual chamber electronic pacemaker",
    "Right bundle branch block, Left anterior fascicular block",
    "Complete right bundle branch block, occasional premature supraventricular complexes",
    "Atrial flutter with variable A-V block"
]
preop_pft_categories = [
    'Normal', 'Mild obstructive', 'Mild restrictive', 'Moderate obstructive',
    'Borderline obstructive', 'Mixed or pure obstructive', 'Severe restrictive',
    'Moderate restrictive', 'Severe obstructive'
]
bmi_category_categories = ['underweight', 'normal', 'overweight', 'obese']
age_group_categories = ['young', 'middle', 'senior', 'elderly']




from pydantic import BaseModel
class PredictionRequest(BaseModel):
    age: float
    sex: str
    bmi: float
    age_group: str
    bmi_category: str
    preop_htn: str
    preop_dm: str
    preop_ecg: str
    preop_pft: str
    preop_hb: float
    preop_plt: float
    preop_pt: float
    preop_aptt: float
    preop_na: float
    preop_k: float
    preop_gluc: float
    preop_alb: float
    preop_ast: float
    preop_alt: float
    preop_bun: float
    preop_cr: float


@app.post("/XGBOOST")
def predict(data: PredictionRequest ):

    # Check categorical values

    # Prepare DataFrame
    df = pd.DataFrame([data.dict()])

    # Derived features
    df['bun_cr_ratio'] = df['preop_bun'] / df['preop_cr']
    df['na_k_ratio'] = df['preop_na'] / df['preop_k']
    df['htn_dm'] = (df['preop_htn'] == 'Y').astype(int) + (df['preop_dm'] == 'Y').astype(int)
    df['hb_by_bun'] = df['preop_hb'] / df['preop_bun']
    df['alb_by_cr'] = df['preop_alb'] / df['preop_cr']
    df['electrolyte_ratio'] = df['preop_na'] / df['preop_k']
    df['gluc_by_bmi'] = df['preop_gluc'] / df['bmi']
    df['hb_by_weight'] = df['preop_hb'] / 75
    df['plt_by_pt'] = df['preop_plt'] / df['preop_pt']

    # Transform
    try:
        X_processed = loaded_preprocessor.transform(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in preprocessing: {str(e)}")

    # Predict
    pred_class = int(loaded_model.predict(X_processed)[0])
    pred_probs = loaded_model.predict_proba(X_processed)[0].tolist()

    # return {
    #     "predicted_class": pred_class ,
    #     "predicted_label": risk_levels.get(pred_class, "Unknown"),
    #     "probabilities": {
    #         risk_levels[i]: prob
    #         for i, prob in enumerate(pred_probs)
    #     }
    # }

    return risk_levels.get(pred_class, "Unknown")



#####################CNN######################

class ECGRequest(BaseModel):
    ecg_signal: List[float]


class EnhancedECG_CNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(EnhancedECG_CNN, self).__init__()

        # First block
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second block
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Third block
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Fourth block
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(2)
        self.dropout4 = nn.Dropout(dropout_rate)

        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Classifier
        self.fc1 = nn.Linear(512, 256)  # 256*2 = 512 (avg + max pooling)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout6 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        # Convolutional layers
        x = self.dropout1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(F.relu(self.bn4(self.conv4(x)))))

        # Global pooling
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        x = torch.cat([avg_pool, max_pool], dim=1)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout5(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout6(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)

        return x


def preprocess_ecg_signal(signal, target_length=5000):
    if len(signal) > target_length:
        # if the signal big i take last part
        processed_signal = signal[-target_length:]
    elif len(signal) < target_length:
        # ا if the signal small repeat it
        repeat_times = target_length // len(signal) + 1
        repeated_signal = np.tile(signal, repeat_times)
        processed_signal = repeated_signal[:target_length]
    else:
        processed_signal = signal

    return processed_signal


# model_dir = r"D:\Projects\Cardiac Patient Monitoring System\models"

# load preprocessing config
with open(fr"{model_dir}\CNN\preprocessing_config.pkl", "rb") as f:
    preprocessing_config = pickle.load(f)

#  load scaler
scaler = joblib.load(fr"{model_dir}\CNN\scaler.pkl")

#  instantiate model and load weights
model = EnhancedECG_CNN(dropout_rate=0.4)
checkpoint = torch.load(fr"{model_dir}\CNN\enhanced_ecg_cnn_model.pth", map_location=torch.device('cpu'), weights_only=False)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


@app.post("/CNN")
def predict_ecg_risk(request: ECGRequest):
    raw_signal = np.array(request.ecg_signal, dtype=float)


    # Preprocessing
    processed_signal = preprocess_ecg_signal(raw_signal, target_length=preprocessing_config['target_length'])
    scaled_signal = scaler.transform([processed_signal])

    with torch.no_grad():
        input_tensor = torch.FloatTensor(scaled_signal).unsqueeze(1)
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        pred_class_idx = np.argmax(probs)
        pred_class_name = preprocessing_config['class_names'][pred_class_idx]

    # return  {
    #     "predicted_class": int(pred_class_idx),
    #     "predicted_label": pred_class_name,
    #     "probabilities": {
    #         preprocessing_config['class_names'][i]: float(probs[i])
    #         for i in range(len(preprocessing_config['class_names']))
    #     }
    # }
    return pred_class_name




##########################LSTM#########################


class Config:
    SIGNAL_NAMES = ["HR", "SBP", "DBP", "MBP", "SpO2", "RR"]
    SEQUENCE_LEN = 60
    HIDDEN_SIZE = 64
    DROPOUT_RATE = 0.3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImprovedLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = torch.mean(attn_out, dim=1)
        output = self.classifier(pooled)
        return output


def predict_on_new_data(new_data: np.ndarray, config: Config):
    device = config.DEVICE

    # Load scaler
    # scaler_path = r"lstm\scaler.pkl"
    scaler = joblib.load(os.path.join(model_dir,r'LSTM\scaler.pkl'))
    new_data_scaled = scaler.transform(new_data)
    new_data_scaled = torch.FloatTensor(new_data_scaled).unsqueeze(0).to(device)

    # Load model
    model = ImprovedLSTMClassifier(
        input_size=len(config.SIGNAL_NAMES),
        hidden_size=config.HIDDEN_SIZE,
        num_classes=2,
        dropout_rate=config.DROPOUT_RATE
    ).to(device)

    # os.path.joint(model_dir, 'lstm\model.pth')
    model_path = os.path.join(model_dir, r'LSTM\model.pth') #"D:\Projects\Cardiac Patient Monitoring System\models\lstm\model.pth"

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(new_data_scaled)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_class = int(np.argmax(probabilities))

    return predicted_class, probabilities.tolist()


# Allow any frontend to connect
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientData(BaseModel):
    data: List[List[float]]  # Should be 60x6 matrix


@app.post("/LSTM")
def predict(data: PatientData):
    config = Config()
    input_array = np.array(data.data)

    predicted_class, probabilities = predict_on_new_data(input_array, config)
    label_map = {0: "Non-Critical Risk", 1: "Critical Risk"}
    label = label_map.get(predicted_class, "Unknown")


    # return {
    #     "predicted_class": int(predicted_class),
    #     "predicted_label": label,
    #     "probabilities": {
    #     label_map.get(idx, f"Class_{idx}"): float(prob)
    #     for idx, prob in enumerate(probabilities)
    # } }
    return label





# ============================================================
# AUTOMATED THREAT PRIORITIZATION — VS CODE / LOCAL VERSION
# ============================================================


from pathlib import Path
import os
from getpass import getpass

# ============================================================
# API KEYS — SAME AS COLAB
# ============================================================

print("Enter your NEW API keys.")
print("Your keys will not be displayed while typing.\n")

vt_key = getpass("VirusTotal API key: ").strip()
abuse_key = getpass("AbuseIPDB API key: ").strip()

os.environ["VIRUSTOTAL_KEY"] = vt_key
os.environ["ABUSEIPDB_KEY"] = abuse_key

print("\nAPI keys loaded into this VS Code session.")

# ============================================================
# IMPORTS — SAME MODELING LIBRARIES AS COLAB
# ============================================================

import re
import json
import time
import random
import ipaddress
import hashlib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

print("Libraries loaded.")
print("TensorFlow:", tf.__version__)

# ============================================================
# LOCAL DATASET SETUP — REPLACES COLAB UPLOAD CELL
# ============================================================

DATASET_FOLDER = Path(r"D:\Desktop\thesisiqraa")

REQUIRED_FILES = [
    DATASET_FOLDER / "AWS_Honeypot_marx-geo.csv",
    DATASET_FOLDER / "london.csv",
    DATASET_FOLDER / "singapore.csv",
]

print("Dataset folder:", DATASET_FOLDER)

missing_files = [str(f) for f in REQUIRED_FILES if not f.exists()]

if missing_files:
    raise FileNotFoundError(
        "The following dataset files are missing:\n"
        + "\n".join(missing_files)
    )

print("All datasets are available.")

# ============================================================
# TERMINAL DISPLAY HELPER
# ============================================================

def display(obj):
    if hasattr(obj, "to_string"):
        print(obj.to_string())
    else:
        print(obj)

# ============================================================
# INSPECT THE ACTUAL DATA — SAME AS COLAB
# ============================================================

for filename in REQUIRED_FILES:
    print("\n" + "=" * 80)
    print(filename.name)
    print("=" * 80)

    sample = pd.read_csv(
        filename,
        nrows=5,
        low_memory=False
    )

    print("Shape of sample:", sample.shape)
    print("\nColumns:")
    print(list(sample.columns))
    print("\nFirst rows:")
    display(sample.head())

# ============================================================
# THREAT DEFINITIONS — SAME AS COLAB
# ============================================================

THREAT_TYPES = [
    "XSS Attempt",
    "Credential Stuffing",
    "Log4j Probe",
    "Port Scan",
    "DDoS Attack",
    "Brute Force",
    "SQL Injection",
    "Malware Link",
]

BASE_SEVERITY = {
    "DDoS Attack": "Critical",
    "Malware Link": "Critical",
    "Credential Stuffing": "High",
    "SQL Injection": "High",
    "Brute Force": "High",
    "XSS Attempt": "Medium",
    "Log4j Probe": "Medium",
    "Port Scan": "Low",
}

SEVERITY_WEIGHT = {
    "Low": 0.25,
    "Medium": 0.50,
    "High": 0.75,
    "Critical": 1.00,
}

# Used only by the priority helper functions later.
SEVERITY_SCORE = {
    "Low": 25,
    "Medium": 50,
    "High": 75,
    "Critical": 100,
}

print("Threat definitions loaded.")

# ============================================================
# PAYLOAD CLASSIFICATION — SAME AS COLAB
# ============================================================

def classify_payload(raw):
    p = str(raw).lower()

    if "jndi:" in p or "${jndi" in p:
        return "Log4j Probe"

    if (
        "<script" in p
        or "onerror=" in p
        or "javascript:" in p
    ):
        return "XSS Attempt"

    if (
        "union select" in p
        or ("select " in p and " from " in p)
    ):
        return "SQL Injection"

    if any(
        x in p
        for x in [
            "shell_exec",
            "new-application",
            "/etc/passwd",
            "base64_decode",
            "wget ",
            "curl ",
        ]
    ):
        return "Malware Link"

    if "eth_blocknumber" in p or "jsonrpc" in p:
        return "Malware Link"

    if "mstshash=" in p:
        return "Brute Force"

    if any(
        x in p
        for x in [
            "login",
            "wp-login",
            "admin",
            "/xmlrpc",
        ]
    ):
        return "Credential Stuffing"

    if "get " in p or "post " in p:
        return "Port Scan"

    return "Port Scan"

PORT_MAP = {
    22: "Brute Force",
    23: "Brute Force",
    3389: "Brute Force",
    445: "Malware Link",
    135: "Malware Link",
    6666: "Malware Link",
    1433: "SQL Injection",
    3306: "SQL Injection",
    53: "DDoS Attack",
    5060: "DDoS Attack",
}

def classify_port(port):
    try:
        port = int(float(port))
    except Exception:
        return "Port Scan"

    return PORT_MAP.get(port, "Port Scan")

print("Classifiers ready.")

# ============================================================
# DATASET LOADING — SAME AS COLAB
# ============================================================

COMMON_COLUMNS = [
    "src",
    "payload_text",
    "type",
    "severity",
    "target",
    "country_code",
    "city",
    "region",
    "latitude",
    "longitude",
    "event_time",
]

def load_payload_dataset(filename, target_cloud):
    df = pd.read_csv(
        filename,
        low_memory=False
    )

    required = ["from", "payload"]

    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"{filename} is missing '{column}'"
            )

    df = df.copy()

    df["src"] = (
        df["from"]
        .astype(str)
        .str.strip()
        .str.strip("'")
        .str.strip('"')
    )

    df["payload_text"] = df["payload"].astype(str)

    df["type"] = df["payload"].apply(
        classify_payload
    )

    df["severity"] = df["type"].map(
        BASE_SEVERITY
    )

    df["target"] = target_cloud

    df["country_code"] = df["country"].astype(str)
    df["city"] = "Unknown"
    df["region"] = "Unknown"
    df["latitude"] = np.nan
    df["longitude"] = np.nan

    if "time" in df.columns:
        df["event_time"] = df["time"].astype(str)
    else:
        df["event_time"] = ""

    return df[COMMON_COLUMNS].copy()

def load_aws_dataset(filename):
    df = pd.read_csv(
        filename,
        on_bad_lines="skip",
        low_memory=False
    )

    required = [
        "srcstr",
        "dpt",
        "cc",
        "country",
        "latitude",
        "longitude",
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"{filename} is missing '{column}'"
            )

    df = df.dropna(
        subset=["srcstr", "dpt"]
    ).copy()

    df["src"] = (
        df["srcstr"]
        .astype(str)
        .str.strip()
    )

    df["type"] = df["dpt"].apply(
        classify_port
    )

    df["severity"] = df["type"].map(
        BASE_SEVERITY
    )

    df["target"] = "AWS"

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    df["country_code"] = df["cc"].astype(str)

    df["city"] = (
        df["locale"]
        .fillna("Unknown")
        .astype(str)
    )

    df["region"] = (
        df["locale"]
        .fillna("Unknown")
        .astype(str)
    )

    df["payload_text"] = (
        "NETWORK PORT: "
        + df["dpt"].astype(str)
    )

    if "datetime" in df.columns:
        df["event_time"] = df["datetime"].astype(str)
    else:
        df["event_time"] = ""

    return df[COMMON_COLUMNS].copy()

# ============================================================
# MULTI-CLOUD PROTOTYPE MAPPING — SAME AS COLAB
# ============================================================

# AWS_Honeypot_marx-geo.csv -> AWS
# london.csv               -> Azure
# singapore.csv            -> GCP

aws = load_aws_dataset(REQUIRED_FILES[0])
london = load_payload_dataset(
    REQUIRED_FILES[1],
    "Azure"
)
singapore = load_payload_dataset(
    REQUIRED_FILES[2],
    "GCP"
)

# ============================================================
# COMBINE
# ============================================================

combined = pd.concat(
    [
        aws,
        london,
        singapore,
    ],
    ignore_index=True
)

# ============================================================
# CLEAN
# ============================================================

combined = combined.dropna(
    subset=[
        "src",
        "payload_text",
        "type",
        "severity",
        "target",
    ]
).copy()

combined["src"] = (
    combined["src"]
    .astype(str)
    .str.strip()
)

# ============================================================
# SAVE COMBINED DATASET
# ============================================================

combined.to_csv(
    DATASET_FOLDER / "combined_threats.csv",
    index=False
)

print("\n" + "=" * 70)
print("MULTI-CLOUD DATASET CREATED")
print("=" * 70)

print("\nCloud distribution:")
display(
    combined["target"].value_counts()
)

print("\nThreat distribution:")
display(
    combined["type"].value_counts()
)

print("\nCoordinates available:")
print(
    combined[
        ["latitude", "longitude"]
    ]
    .notna()
    .all(axis=1)
    .sum()
)

print("\nTotal records:")
print(len(combined))

print("\nUnique IPs:")
print(combined["src"].nunique())

display(combined.head())

# ============================================================
# VALIDATE DATA
# ============================================================

print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print("\nTotal records:", len(combined))
print(
    "Unique IP addresses:",
    combined["src"].nunique()
)

print("\nThreat distribution:")
display(
    combined["type"].value_counts()
)

print("\nSeverity distribution:")
display(
    combined["severity"].value_counts()
)

print("\nMissing values:")
display(
    combined.isnull().sum()
)

print("\ncombined_threats.csv already saved after cleaning.")

# ============================================================
# FEATURE ENGINEERING — SAME AS COLAB
# ============================================================

def ip_to_features(ip):
    try:
        parts = str(ip).split(".")

        if len(parts) != 4:
            return [0, 0, 0, 0]

        values = [int(x) for x in parts]

        if not all(
            0 <= x <= 255
            for x in values
        ):
            return [0, 0, 0, 0]

        return values

    except Exception:
        return [0, 0, 0, 0]

def payload_to_features(payload):
    p = str(payload).lower()

    return [
        int("<script" in p),
        int("jndi" in p),
        int("union select" in p),
        int(
            "login" in p
            or "wp-login" in p
        ),
        int("mstshash" in p),
        int(
            "wget" in p
            or "curl" in p
        ),
        int(
            "jsonrpc" in p
            or "eth_blocknumber" in p
        ),
        int(
            "select " in p
            and " from " in p
        ),
        int("network port" in p),
    ]

def encode_row(row):
    ip_features = ip_to_features(
        row["src"]
    )

    payload_features = payload_to_features(
        row["payload_text"]
    )

    cloud_features = [
        int(row["target"] == "AWS"),
        int(row["target"] == "Azure"),
        int(row["target"] == "GCP"),
    ]

    return (
        ip_features
        + payload_features
        + cloud_features
    )

# ============================================================
# CREATE PURE NUMPY ARRAYS FOR MACHINE LEARNING
# ============================================================
# The CSV files can be read by pandas using a PyArrow-backed dtype.
# We explicitly convert everything to ordinary NumPy arrays here so
# scikit-learn never receives a pandas/Arrow ExtensionArray.

X = np.array(
    [
        encode_row(row)
        for _, row in combined.iterrows()
    ],
    dtype=np.float32,
    copy=True
)

# Convert through a normal Python list first. This completely removes
# any pandas/pyarrow array wrapper from the target values.
y = np.array(
    combined["type"].astype(str).tolist(),
    dtype=str
)

# Final safety conversion.
X = np.array(X, dtype=np.float32, copy=True)
y = np.array(y, dtype=str, copy=True)

print("X type:", type(X))
print("y type:", type(y))
print("X dtype:", X.dtype)
print("y dtype:", y.dtype)
print("X shape:", X.shape)
print("y shape:", y.shape)

# ============================================================
# TRAIN / TEST SPLIT — SAME COLAB SETTINGS
# ============================================================
# Split the row indices rather than passing X/y directly to
# train_test_split. This avoids the pandas/pyarrow indexing error
# seen on Windows while keeping exactly the same 80/20 split,
# random_state=42 and stratification used in the Colab workflow.

row_indices = np.arange(len(X), dtype=np.int64)

# IMPORTANT: build stratification labels as a plain Python/NumPy array.
# pandas with a PyArrow string backend can otherwise reach scikit-learn
# and cause: "only integer scalar arrays can be converted to a scalar index".
stratify_labels = np.array(
    [str(value) for value in combined["type"].tolist()],
    dtype=object
)

train_indices, test_indices = train_test_split(
    row_indices,
    test_size=0.20,
    random_state=42,
    stratify=stratify_labels
)

# Use only ordinary NumPy indexing from this point onward.
X_train = np.array(X[train_indices], dtype=np.float32, copy=True)
X_test = np.array(X[test_indices], dtype=np.float32, copy=True)
y_train = np.array(y[train_indices], dtype=str, copy=True)
y_test = np.array(y[test_indices], dtype=str, copy=True)

print(
    "Training records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)

# ============================================================
# MODEL 1 — RANDOM FOREST
# EXACT COLAB SETTINGS
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=250,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(
    X_test
)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_precision = precision_score(
    y_test,
    rf_pred,
    average="weighted",
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_pred,
    average="weighted",
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

print(
    f"Accuracy : {rf_accuracy:.4f}"
)

print(
    f"Precision: {rf_precision:.4f}"
)

print(
    f"Recall   : {rf_recall:.4f}"
)

print(
    f"F1 Score : {rf_f1:.4f}"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_pred,
        zero_division=0
    )
)

# ============================================================
# RANDOM FOREST CONFUSION MATRIX
# ============================================================

labels = sorted(
    combined["type"].unique()
)

cm = confusion_matrix(
    y_test,
    rf_pred,
    labels=labels
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(
    ax=ax,
    colorbar=False,
    xticks_rotation=45
)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    DATASET_FOLDER
    / "random_forest_confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    "Random Forest confusion matrix saved."
)

# ============================================================
# CHECK ACTUAL THREAT CLASSES
# ============================================================

print(
    "\nThreat classes in COMPLETE dataset:"
)

print(
    sorted(
        combined["type"].unique()
    )
)

print(
    "\nNumber of classes in complete dataset:",
    combined["type"].nunique()
)

print("\nClass distribution:")
print(
    combined["type"].value_counts()
)

print("\nClasses in training data:")
print(
    sorted(
        np.unique(y_train)
    )
)

print("\nClasses in test data:")
print(
    sorted(
        np.unique(y_test)
    )
)

# ============================================================
# MODEL 2 — NEURAL NETWORK
# FINAL COLAB CONFIGURATION
# ============================================================

X_train = np.asarray(
    X_train,
    dtype=np.float64
)

X_test = np.asarray(
    X_test,
    dtype=np.float64
)

if not np.isfinite(X_train).all():
    raise ValueError(
        "X_train contains NaN or infinite values."
    )

if not np.isfinite(X_test).all():
    raise ValueError(
        "X_test contains NaN or infinite values."
    )

y_train_nn = np.asarray(
    y_train,
    dtype=str
).ravel()

y_test_nn = np.asarray(
    y_test,
    dtype=str
).ravel()

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "Classes:",
    np.unique(y_train_nn)
)

nn_scaler = StandardScaler()

X_train_nn = nn_scaler.fit_transform(
    X_train
)

X_test_nn = nn_scaler.transform(
    X_test
)

nn_model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    learning_rate_init=0.001,
    batch_size=2048,
    max_iter=100,
    early_stopping=False,
    random_state=42,
    verbose=True
)

print(
    "\nStarting Neural Network training..."
)

nn_model.fit(
    X_train_nn,
    y_train_nn
)

nn_pred = nn_model.predict(
    X_test_nn
)

nn_accuracy = accuracy_score(
    y_test_nn,
    nn_pred
)

nn_precision = precision_score(
    y_test_nn,
    nn_pred,
    average="weighted",
    zero_division=0
)

nn_recall = recall_score(
    y_test_nn,
    nn_pred,
    average="weighted",
    zero_division=0
)

nn_f1 = f1_score(
    y_test_nn,
    nn_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 65)
print("NEURAL NETWORK RESULTS")
print("=" * 65)

print(
    f"Accuracy : {nn_accuracy:.4f}"
)

print(
    f"Precision: {nn_precision:.4f}"
)

print(
    f"Recall   : {nn_recall:.4f}"
)

print(
    f"F1 Score : {nn_f1:.4f}"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test_nn,
        nn_pred,
        zero_division=0
    )
)

# ============================================================
# MODEL 3 — LSTM DATA PREPARATION
# SAME AS COLAB
# ============================================================

lstm_encoder = LabelEncoder()

combined["threat_encoded"] = (
    lstm_encoder.fit_transform(
        combined["type"]
    )
)

combined["severity_numeric"] = (
    combined["severity"]
    .map(SEVERITY_WEIGHT)
)

combined["cloud_numeric"] = (
    combined["target"]
    .map(
        {
            "AWS": 0,
            "Azure": 1,
            "GCP": 2,
        }
    )
)

lstm_data = combined[
    [
        "threat_encoded",
        "severity_numeric",
        "cloud_numeric",
    ]
].values.astype(
    np.float32
)

lstm_scaler = StandardScaler()

lstm_data_scaled = (
    lstm_scaler.fit_transform(
        lstm_data
    )
)

TIMESTEPS = 8

X_lstm = []
y_lstm = []

for i in range(
    TIMESTEPS,
    len(lstm_data_scaled)
):
    X_lstm.append(
        lstm_data_scaled[
            i - TIMESTEPS:i
        ]
    )

    y_lstm.append(
        combined[
            "threat_encoded"
        ].iloc[i]
    )

X_lstm = np.array(
    X_lstm,
    dtype=np.float32
)

y_lstm = np.array(
    y_lstm,
    dtype=np.int32
)

print(
    "LSTM input:",
    X_lstm.shape
)

print(
    "LSTM target:",
    y_lstm.shape
)

# ============================================================
# LSTM TRAIN / TEST
# SAME AS COLAB
# ============================================================

lstm_split = int(
    len(X_lstm) * 0.80
)

X_lstm_train = X_lstm[
    :lstm_split
]

X_lstm_test = X_lstm[
    lstm_split:
]

y_lstm_train = y_lstm[
    :lstm_split
]

y_lstm_test = y_lstm[
    lstm_split:
]

print(
    "LSTM training:",
    X_lstm_train.shape
)

print(
    "LSTM testing:",
    X_lstm_test.shape
)

# ============================================================
# BUILD LSTM
# SAME AS COLAB
# ============================================================

num_classes = len(
    lstm_encoder.classes_
)

lstm_model = Sequential(
    [
        Input(
            shape=(
                TIMESTEPS,
                X_lstm_train.shape[2],
            )
        ),

        LSTM(
            64,
            return_sequences=True
        ),

        Dropout(0.25),

        LSTM(32),

        Dropout(0.25),

        Dense(
            32,
            activation="relu"
        ),

        Dense(
            num_classes,
            activation="softmax"
        ),
    ]
)

lstm_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

lstm_model.summary()

# ============================================================
# TRAIN LSTM
# SAME AS COLAB
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = lstm_model.fit(
    X_lstm_train,
    y_lstm_train,
    validation_split=0.15,
    epochs=30,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

print(
    "LSTM training finished."
)

# ============================================================
# LSTM EVALUATION
# ============================================================

lstm_loss, lstm_accuracy = (
    lstm_model.evaluate(
        X_lstm_test,
        y_lstm_test,
        verbose=0
    )
)

lstm_probabilities = (
    lstm_model.predict(
        X_lstm_test,
        verbose=0
    )
)

lstm_prediction_encoded = (
    np.argmax(
        lstm_probabilities,
        axis=1
    )
)

lstm_prediction = (
    lstm_encoder.inverse_transform(
        lstm_prediction_encoded
    )
)

lstm_actual = (
    lstm_encoder.inverse_transform(
        y_lstm_test
    )
)

lstm_precision = precision_score(
    lstm_actual,
    lstm_prediction,
    average="weighted",
    zero_division=0
)

lstm_recall = recall_score(
    lstm_actual,
    lstm_prediction,
    average="weighted",
    zero_division=0
)

lstm_f1 = f1_score(
    lstm_actual,
    lstm_prediction,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 60)
print("LSTM")
print("=" * 60)

print(
    f"Accuracy : {lstm_accuracy:.4f}"
)

print(
    f"Precision: {lstm_precision:.4f}"
)

print(
    f"Recall   : {lstm_recall:.4f}"
)

print(
    f"F1 Score : {lstm_f1:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        lstm_actual,
        lstm_prediction,
        zero_division=0
    )
)

# ============================================================
# THREE-MODEL COMPARISON
# ============================================================

model_results = pd.DataFrame(
    {
        "Model": [
            "Random Forest",
            "Neural Network",
            "LSTM",
        ],

        "Accuracy": [
            rf_accuracy,
            nn_accuracy,
            lstm_accuracy,
        ],

        "Precision": [
            rf_precision,
            nn_precision,
            lstm_precision,
        ],

        "Recall": [
            rf_recall,
            nn_recall,
            lstm_recall,
        ],

        "F1 Score": [
            rf_f1,
            nn_f1,
            lstm_f1,
        ],
    }
)

print("\n" + "=" * 70)
print("THREE MODEL COMPARISON")
print("=" * 70)
print(model_results.to_string(index=False))

model_results.to_csv(
    DATASET_FOLDER / "model_results.csv",
    index=False
)

# ============================================================
# K-MEANS IP BEHAVIOUR
# SAME AS COLAB
# ============================================================

combined["severity_numeric"] = (
    combined["severity"]
    .map(SEVERITY_WEIGHT)
)

ip_stats = (
    combined
    .groupby("src")
    .agg(
        attack_count=(
            "src",
            "count"
        ),

        average_severity=(
            "severity_numeric",
            "mean"
        )
    )
    .reset_index()
)

cluster_scaler = StandardScaler()

cluster_X = (
    cluster_scaler.fit_transform(
        ip_stats[
            [
                "attack_count",
                "average_severity",
            ]
        ]
    )
)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

ip_stats["cluster"] = (
    kmeans.fit_predict(
        cluster_X
    )
)

print(
    "\nK-Means trained."
)

print(
    ip_stats.head().to_string(
        index=False
    )
)

# ============================================================
# PUBLIC-IP VALIDATION
# ============================================================

def valid_public_ip(ip):
    try:
        addr = ipaddress.ip_address(
            str(ip).strip()
        )

        return (
            addr.is_global
            and not addr.is_private
            and not addr.is_loopback
            and not addr.is_reserved
            and not addr.is_multicast
        )

    except ValueError:
        return False

# ============================================================
# ABUSEIPDB API
# SAME AS COLAB
# ============================================================

def abuseipdb_lookup(ip):
    api_key = os.getenv(
        "ABUSEIPDB_KEY"
    )

    if not api_key:
        return {
            "success": False,
            "error": "AbuseIPDB API key missing."
        }

    if not valid_public_ip(ip):
        return {
            "success": False,
            "error": "Not a valid public IP."
        }

    url = (
        "https://api.abuseipdb.com/api/v2/check"
    )

    try:
        response = requests.get(
            url,
            headers={
                "Accept": "application/json",
                "Key": api_key,
            },
            params={
                "ipAddress": ip,
                "maxAgeInDays": 90,
            },
            timeout=15
        )

        if response.status_code != 200:
            return {
                "success": False,
                "status_code": response.status_code,
                "error": response.text[:500],
            }

        data = (
            response
            .json()
            .get("data", {})
        )

        return {
            "success": True,
            "ip": data.get(
                "ipAddress",
                ip
            ),
            "abuse_confidence": data.get(
                "abuseConfidenceScore",
                0
            ),
            "country": data.get(
                "countryCode",
                "Unknown"
            ),
            "isp": data.get(
                "isp",
                "Unknown"
            ),
            "domain": data.get(
                "domain",
                "Unknown"
            ),
            "total_reports": data.get(
                "totalReports",
                0
            ),
            "last_reported": data.get(
                "lastReportedAt",
                None
            ),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

print(
    "AbuseIPDB function ready."
)

# ============================================================
# FINAL PRIORITY HELPER
# SAME PROJECT LOGIC
# ============================================================

def calculate_priority(
    rf_score,
    nn_score,
    lstm_score,
    severity,
    abuse_result=None
):
    ai_score = (
        rf_score * 25
        + nn_score * 20
        + lstm_score * 20
        + SEVERITY_SCORE.get(
            severity,
            50
        ) * 0.35
    )

    final_score = ai_score

    if (
        abuse_result
        and abuse_result.get("success")
    ):
        abuse_confidence = float(
            abuse_result.get(
                "abuse_confidence",
                0
            )
        )

        final_score += (
            abuse_confidence * 0.15
        )

    final_score = max(
        0,
        min(
            final_score,
            100
        )
    )

    if final_score >= 80:
        priority = "CRITICAL"
    elif final_score >= 60:
        priority = "HIGH"
    elif final_score >= 35:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    return (
        round(final_score, 2),
        priority,
        round(ai_score, 2)
    )

# ============================================================
# FINAL PRIORITY SCORE
# ============================================================

def calculate_final_priority(
    rf_score,
    nn_score,
    lstm_score,
    severity,
    abuse_result=None,
    attack_count=1
):
    model_confidence = (
        rf_score
        + nn_score
        + lstm_score
    ) / 3

    severity_score = (
        SEVERITY_SCORE.get(
            severity,
            50
        )
    )

    if (
        abuse_result
        and abuse_result.get("success")
    ):
        abuse_score = float(
            abuse_result.get(
                "abuse_confidence",
                0
            )
        )
    else:
        abuse_score = 0

    behaviour_score = min(
        attack_count * 10,
        100
    )

    final_score = (
        model_confidence * 0.40
        + severity_score * 0.25
        + abuse_score * 0.20
        + behaviour_score * 0.15
    )

    final_score = max(
        0,
        min(
            final_score,
            100
        )
    )

    if final_score >= 80:
        priority = "CRITICAL"
    elif final_score >= 60:
        priority = "HIGH"
    elif final_score >= 35:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    return (
        round(final_score, 2),
        priority
    )

# ============================================================
# SAVE MODELS
# ============================================================

joblib.dump(
    rf_model,
    DATASET_FOLDER / "random_forest.pkl"
)

joblib.dump(
    nn_model,
    DATASET_FOLDER / "neural_network.pkl"
)

joblib.dump(
    nn_scaler,
    DATASET_FOLDER / "neural_network_scaler.pkl"
)

joblib.dump(
    lstm_encoder,
    DATASET_FOLDER / "lstm_label_encoder.pkl"
)

joblib.dump(
    lstm_scaler,
    DATASET_FOLDER / "lstm_scaler.pkl"
)

joblib.dump(
    kmeans,
    DATASET_FOLDER / "kmeans.pkl"
)

joblib.dump(
    cluster_scaler,
    DATASET_FOLDER / "cluster_scaler.pkl"
)

lstm_model.save(
    DATASET_FOLDER / "lstm_model.keras"
)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Random Forest Accuracy : {rf_accuracy:.4f}"
)

print(
    f"Neural Network Accuracy: {nn_accuracy:.4f}"
)

print(
    f"LSTM Accuracy          : {float(lstm_accuracy):.4f}"
)

print("\nSaved artefacts in:")
print(DATASET_FOLDER)

print("\nModel files:")
print("random_forest.pkl")
print("neural_network.pkl")
print("neural_network_scaler.pkl")
print("lstm_model.keras")
print("lstm_label_encoder.pkl")
print("lstm_scaler.pkl")
print("kmeans.pkl")
print("cluster_scaler.pkl")
print("combined_threats.csv")
print("model_results.csv")
print("random_forest_confusion_matrix.png")

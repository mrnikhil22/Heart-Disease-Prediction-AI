import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ===============================
# LOAD DATASET
# ===============================
print("Loading Dataset...")

data = pd.read_csv("heart.csv")

X = data.drop("target", axis=1)
y = data["target"]

# ===============================
# LOAD SCALER
# ===============================
print("Loading Scaler...")

scaler = joblib.load("scaler.pkl")

X_scaled = scaler.transform(X)

# ===============================
# TRAIN TEST SPLIT
# (Same random_state as training)
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ===============================
# LOAD MODEL
# ===============================

print("Loading AI Model...")

model = load_model("mymodel.keras")

# ===============================
# MAKE PREDICTIONS
# ===============================

predictions = model.predict(X_test)

y_pred = (predictions > 0.5).astype(int)

# ===============================
# CALCULATE METRICS
# ===============================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ===============================
# PRINT RESULTS
# ===============================

print("\n====================================")
print(" HEART DISEASE MODEL PERFORMANCE")
print("====================================")

print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1 Score : {f1*100:.2f}%")

print("\n====================================")
print("CLASSIFICATION REPORT")
print("====================================")

print(classification_report(y_test, y_pred))

print("\n====================================")
print("CONFUSION MATRIX")
print("====================================")

print(confusion_matrix(y_test, y_pred))
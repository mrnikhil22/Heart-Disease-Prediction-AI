"""Train the 13-16-8-1 ANN on REAL UCI Cleveland data with the corrected protocol.

Data: UCI Heart Disease, Cleveland subset (Detrano et al., 1988).
303 rows; rows with missing ca/thal dropped -> 297 (160 no disease, 137 disease).
Source: https://archive.ics.uci.edu/ml/datasets/heart+Disease

Protocol (fixes the original pipeline, see paper Section VII):
  1. stratified 80/20 split FIRST (random_state 42)
  2. StandardScaler fitted on TRAINING rows only
  3. train 50 epochs, batch 10 (same as before)
  4. evaluate on the held-out test set
Outputs: mymodel.keras, scaler.pkl (same names the Flask app expects).
"""
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

SEED = 42
np.random.seed(SEED)

data = pd.read_csv("heart.csv")          # real Cleveland data, 297 rows
X = data.drop("target", axis=1)
y = data["target"]
if X.shape[1] != 13:
    raise Exception(f"Expected 13 features but got {X.shape[1]}")

# 1. split BEFORE any fitting (stratified keeps the 160/137 ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# 2. scaler sees training rows only
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# 3. same architecture as before
model = Sequential()
model.add(Dense(16, input_dim=13, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])
model.fit(X_train_s, y_train, epochs=50, batch_size=10, verbose=1)

# 4. held-out evaluation
p = model.predict(X_test_s).ravel()
y_pred = (p > 0.5).astype(int)
print("\n==== HELD-OUT TEST RESULTS (real Cleveland data) ====")
print(f"Accuracy : {accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"Precision: {precision_score(y_test, y_pred)*100:.1f}%")
print(f"Recall   : {recall_score(y_test, y_pred)*100:.1f}%")
print(f"F1 Score : {f1_score(y_test, y_pred)*100:.1f}%")
print(f"AUC      : {roc_auc_score(y_test, p):.3f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

model.save("mymodel.keras")
print("\nSaved mymodel.keras and scaler.pkl (trained on real data).")

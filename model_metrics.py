"""Re-evaluate the saved model on the held-out test set (no training here).

Splits heart.csv the same way as model_training.py (stratified, seed 42),
loads the saved scaler and Keras model, and prints the metrics.
"""
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

SEED = 42

data = pd.read_csv("heart.csv")
X = data.drop("target", axis=1)
y = data["target"]

# identical split to training: the test rows were never used for fitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

scaler = joblib.load("scaler.pkl")
X_test_s = scaler.transform(X_test)

model = load_model("mymodel.keras")
p = model.predict(X_test_s).ravel()
y_pred = (p > 0.5).astype(int)

print("\n====================================")
print(" HEART DISEASE MODEL PERFORMANCE")
print(" (real UCI Cleveland data, held-out)")
print("====================================")
print(f"Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred)*100:.2f}%")
print(f"Recall   : {recall_score(y_test, y_pred)*100:.2f}%")
print(f"F1 Score : {f1_score(y_test, y_pred)*100:.2f}%")
print(f"AUC      : {roc_auc_score(y_test, p):.3f}")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))

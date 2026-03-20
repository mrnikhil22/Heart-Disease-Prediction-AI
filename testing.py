import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore

test_input = pd.read_csv("testing_input_dataset.csv")
x_test = test_input.to_numpy()
test_output = pd.read_csv("testing_output_dataset.csv")
y_test = test_output.to_numpy()

model = load_model("mymodel.keras")

predictions = model.predict(x_test)
y_pred = ((predictions > 0.5).astype(int))
y_test = ((y_test > 0.5).astype(int))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy report:")
print(f"Correctly predicted {np.sum(y_pred == y_test)} out of {len(y_test)} samples")
print("Accuracy:", accuracy*100 , "%")
print()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix: \n{cm}")
print(f"{cm[0,0]} True Negatives : Model correctly predicted no disease")
print(f"{cm[0,1]} False Positives : Model predicted “disease”/1 but patient actually had no disease (false alarm)")
print(f"{cm[1,0]} False Negatives : Model predicted “no disease”/0 but patient actually had disease (missed case)")
print(f"{cm[1,1]} True Positives : Model correctly predicted disease")
print()

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(f"Classificstion report:\n{report}") 
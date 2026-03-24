import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import joblib

# ✅ LOAD DATA
data = pd.read_csv("heart.csv")

# ✅ CHECK (IMPORTANT DEBUG)
print("Columns:", data.columns)
print("Shape:", data.shape)

# ✅ SPLIT INPUT & OUTPUT
X = data.drop("target", axis=1)
y = data["target"]

# ✅ ENSURE 13 FEATURES
if X.shape[1] != 13:
    raise Exception(f"❌ ERROR: Expected 13 features but got {X.shape[1]}")

# ✅ SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ SAVE SCALER
joblib.dump(scaler, "scaler.pkl")

# ✅ MODEL (AUTO 13 INPUT)
model = Sequential()
model.add(Dense(16, input_dim=13, activation='relu'))   # ✅ FIXED
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ✅ TRAIN
model.fit(X_scaled, y, epochs=50, batch_size=10)

# ✅ SAVE MODEL
model.save("mymodel.keras")

print("✅ Training Complete (13 Features Model Ready)")
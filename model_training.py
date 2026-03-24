import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
X = pd.read_csv("training_input_dataset.csv")
y = pd.read_csv("training_output_dataset.csv")

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Model
model = Sequential()
model.add(Dense(16, input_dim=12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X_scaled, y, epochs=50, batch_size=10)

# Save model
model.save("mymodel.keras")

print("Training Complete ✅")
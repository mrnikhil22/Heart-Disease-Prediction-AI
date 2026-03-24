import pandas as pd
import numpy as np

# Seed for reproducibility
np.random.seed(42)

# Number of rows per risk level
n_low = 150
n_medium = 150
n_high = 150

# Helper function to generate realistic continuous features
def generate_continuous(low, high, n):
    return np.round(np.random.uniform(low, high, n), 1)

# ---------------- LOW RISK ----------------
low_data = pd.DataFrame({
    'age': generate_continuous(25, 35, n_low),
    'sex': np.random.choice([0,1], n_low),
    'cp': np.random.choice([0,1], n_low),
    'trestbps': generate_continuous(110, 130, n_low),
    'chol': generate_continuous(150, 200, n_low),
    'fbs': np.zeros(n_low, dtype=int),
    'restecg': np.random.choice([0,1], n_low),
    'thalach': generate_continuous(160, 190, n_low),
    'exang': np.zeros(n_low, dtype=int),
    'oldpeak': generate_continuous(0, 1, n_low),
    'slope': np.random.choice([1,2], n_low),
    'ca': np.zeros(n_low, dtype=int),
    'thal': np.random.choice([1,2], n_low),
    'target': np.zeros(n_low, dtype=int)  # 0 = Low Risk
})

# ---------------- MEDIUM RISK ----------------
medium_data = pd.DataFrame({
    'age': generate_continuous(36, 50, n_medium),
    'sex': np.random.choice([0,1], n_medium),
    'cp': np.random.choice([1,2], n_medium),
    'trestbps': generate_continuous(130, 150, n_medium),
    'chol': generate_continuous(200, 250, n_medium),
    'fbs': np.random.choice([0,1], n_medium),
    'restecg': np.random.choice([0,1], n_medium),
    'thalach': generate_continuous(140, 160, n_medium),
    'exang': np.random.choice([0,1], n_medium),
    'oldpeak': generate_continuous(1, 2.5, n_medium),
    'slope': np.random.choice([1,2], n_medium),
    'ca': np.random.choice([0,1], n_medium),
    'thal': np.random.choice([2,3], n_medium),
    'target': np.ones(n_medium, dtype=int)  # 1 = Medium Risk
})

# ---------------- HIGH RISK ----------------
high_data = pd.DataFrame({
    'age': generate_continuous(51, 65, n_high),
    'sex': np.random.choice([0,1], n_high),
    'cp': np.random.choice([2,3], n_high),
    'trestbps': generate_continuous(150, 180, n_high),
    'chol': generate_continuous(250, 320, n_high),
    'fbs': np.ones(n_high, dtype=int),
    'restecg': np.random.choice([1,2], n_high),
    'thalach': generate_continuous(110, 140, n_high),
    'exang': np.ones(n_high, dtype=int),
    'oldpeak': generate_continuous(2.5, 4, n_high),
    'slope': np.random.choice([0,1], n_high),
    'ca': np.random.choice([1,2,3], n_high),
    'thal': np.random.choice([2,3], n_high),
    'target': np.ones(n_high, dtype=int)  # 1 = High Risk
})

# Combine all data
heart_data = pd.concat([low_data, medium_data, high_data], ignore_index=True)

# Shuffle rows
heart_data = heart_data.sample(frac=1).reset_index(drop=True)

# Save to CSV
heart_data.to_csv('heart.csv', index=False)

print('Balanced heart.csv with 450 rows generated successfully ✅')
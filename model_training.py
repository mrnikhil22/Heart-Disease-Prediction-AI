import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pandas as pd

def neural_network(x, y):
    model = Sequential([
        Dense(12, activation='relu', input_shape=(12,)),  # 👈 FIXED
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy()
    )

    model.fit(x, y, epochs=100)  # 👈 reduce epochs (fast training)
    
    model.save("mymodel.keras")  # 👈 correct save

def main():
    input_data = pd.read_csv("training_input_dataset.csv")
    x = input_data.to_numpy()

    output_data = pd.read_csv("training_output_dataset.csv")
    y = output_data.to_numpy()

    neural_network(x, y)

main()
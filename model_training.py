import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pandas as pd

def neural_network(x, y):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(x)  # Learn the mean and variance from the input data
    model = Sequential([
        normalizer,  # Add the normalization layer as the first input layer
        Dense(12, activation='relu', name='layer1'),   # First hidden layer with 12 units
        Dense(16, activation='relu', name='layer2'),   # Second hidden layer with 16 units
        Dense(8, activation='relu', name='layer3'),    # Third hidden layer with 8 units
        Dense(1, activation='sigmoid', name='layer4')  # Output layer (sigmoid for binary classification)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy()
    )
    model.fit(x, y, epochs=500) # Train the model on the dataset for 500 epochs
    model.save("mymodel.keras")
    
def main():
    input = pd.read_csv("training_input_dataset.csv")
    x = input.to_numpy()  # Convert to NumPy array
    output = pd.read_csv("training_output_dataset.csv")
    y = output.to_numpy()  # Convert to NumPy array
    neural_network(x, y)

main()

# Install required libraries if not already installed
# pip install tensorflow pandas scikit-learn h5py

import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv(r'C:\Dev\working AI script\cancer.csv')

# Explore the data
print(data.head())
print(data.info())
print(data.describe())

# Prepare the data for training
X = data.drop(columns=['diagnosis(1=m, 0=b)'])
y = data['diagnosis(1=m, 0=b)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=1000, validation_split=0.2, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.5f}")

# Predict and calculate accuracy
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.5f}")

# Save the model
model.save('cancer_model.h5')

# Interact with the saved HDF5 file using h5py
with h5py.File('cancer_model.h5', 'r') as f:
    # List all groups in the file
    print("Keys: %s" % f.keys())

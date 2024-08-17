# model.py
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the provided CSV file to create a mock training scenario for model building
csv_path = 'cancer.csv'
data = pd.read_csv(csv_path)

# Split data into features and target
X = data.drop(columns=['diagnosis(1=m, 0=b)'])
y = data['diagnosis(1=m, 0=b)']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model (using fewer epochs for this example)
model.fit(X_scaled, y, epochs=10, batch_size=32, validation_split=0.2)

# Save the model
model.save('cancer_model.h5')

def predict(features):
    """
    Predicts whether the tumor is malignant or benign.
    
    Args:
    features (np.array): The input features for prediction.
    
    Returns:
    str: 'Malignant' or 'Benign'
    """
    prediction = model.predict(features)
    return 'Malignant' if prediction[0] > 0.5 else 'Benign'

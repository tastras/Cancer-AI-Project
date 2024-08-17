# nlp_processing.py
import spacy
import numpy as np

# Load the NLP model
nlp = spacy.load("en_core_web_sm")

def extract_features_from_text(text, expected_num_features):
    """
    Extracts features from natural language text.
    
    Args:
    text (str): The input text containing the features.
    expected_num_features (int): The number of features the model expects.
    
    Returns:
    np.array: A numpy array containing the extracted features.
    """
    doc = nlp(text)
    
    # Extracting numerical values from the text
    features = [float(token.text) for token in doc if token.like_num]
    
    # Ensure the extracted features match the expected input shape
    if len(features) != expected_num_features:
        raise ValueError(f"Expected {expected_num_features} features, but got {len(features)}")
    
    return np.array(features).reshape(1, -1)

import numpy as np
import torch
from predictor_model import predict
import random

# Define colors for different models
PREDICTOR_COLORS = {
    'delta_20': (255, 0, 0),  # Red
    'delta_30': (255, 128, 0),  # Orange
    'delta_40': (255, 255, 0),  # Yellow
    'delta_50': (0, 255, 0),  # Green
    'delta_60': (0, 255, 128),  # Light Green
    'delta_75': (0, 0, 255),  # Blue
}

def get_random_color(used_colors):
    # Generate a random non-white, non-black color that is not already used
    while True:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if color != (255, 255, 255) and color != (0, 0, 0) and color not in used_colors:
            return color

def predictor_delta(points, model, seq_length, output_size=1, normalize=False):
    if len(points) < seq_length:
        return None
    
    if normalize:
        input_data = np.array(points[-seq_length:]) / np.array([1000, 1000])
    else:
        input_data = np.array(points[-seq_length:])
    
    input_data = input_data.flatten()
    input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
    prediction = predict(model, input_tensor)
    
    # Reshape the prediction to the desired format
    prediction = prediction.reshape(-1, 2).tolist()
    return prediction

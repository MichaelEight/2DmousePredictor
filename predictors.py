import numpy as np
import torch
from ml_model import predict
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

def predictor_delta(points, model, seq_length, normalize=False):
    if len(points) < seq_length:
        return None
    
    input_data = np.array(points[-seq_length:])
    if normalize:
        input_data[:, 0] /= 1000
        input_data[:, 1] /= 1000
    input_data = input_data.flatten()
    input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
    prediction = predict(model, input_tensor)
    
    return tuple(prediction[0].tolist())


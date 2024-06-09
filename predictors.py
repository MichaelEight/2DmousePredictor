import numpy as np
import torch
from ml_model import MousePredictor, predict

# Define colors for different models
PREDICTOR_COLORS = {
    'delta_20': (255, 0, 0),  # Red
    'delta_50': (0, 255, 0),  # Green
    #'delta_75': (0, 0, 255),  # Blue
    # Add more colors if you have more models
}

def predictor_delta(points, model, seq_length):
    if len(points) < seq_length:
        return None
    
    input_data = np.array(points[-seq_length:]).flatten()
    input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
    prediction = predict(model, input_tensor)
    
    return tuple(prediction[0].tolist())

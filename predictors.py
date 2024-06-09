import numpy as np
import torch
from ml_model import predict

PREDICTOR_COLORS = {
    "delta": (255, 255, 0)  # Yellow
}

def predictor_delta(points, model, sequence_length):
    if len(points) < sequence_length:
        return None
    
    input_data = np.array(points[-sequence_length:]).flatten()
    input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
    prediction = predict(model, input_tensor)
    
    return tuple(prediction[0].tolist())

import numpy as np
import torch
from ml_model import predict

PREDICTOR_COLORS = {
    "delta": (255, 255, 0)  # Yellow
}

def predictor_delta(points, model):
    if len(points) < 20:
        return None
    
    input_data = np.array(points[-20:]).flatten()
    input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
    prediction = predict(model, input_tensor)
    
    return tuple(prediction[0].tolist())

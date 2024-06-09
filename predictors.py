import numpy as np
import torch
from ml_model import MousePredictor, predict

PREDICTOR_COLORS = {
    "alpha": (255, 0, 0),  # Red
    "beta": (0, 255, 0),   # Green
    "gamma": (0, 0, 255),  # Blue
    "delta": (255, 255, 0)  # Yellow
}

def predictor_alpha(points):
    if len(points) < 2:
        return None
    
    dx = points[-1][0] - points[-2][0]
    dy = points[-1][1] - points[-2][1]
    
    next_x = points[-1][0] + dx
    next_y = points[-1][1] + dy
    
    return (next_x, next_y)

def predictor_beta(points):
    if len(points) < 3:
        return None
    
    dx1 = points[-1][0] - points[-2][0]
    dy1 = points[-1][1] - points[-2][1]
    
    dx2 = points[-2][0] - points[-3][0]
    dy2 = points[-2][1] - points[-3][1]
    
    ddx = dx1 - dx2
    ddy = dy1 - dy2
    
    next_dx = dx1 + ddx
    next_dy = dy1 + ddy
    
    next_x = points[-1][0] + next_dx
    next_y = points[-1][1] + next_dy
    
    return (next_x, next_y)

def predictor_gamma(points):
    if len(points) < 4:
        return None
    
    positions = np.array(points)
    
    # Calculate velocity
    velocities = np.diff(positions, axis=0)
    
    # Calculate acceleration
    accelerations = np.diff(velocities, axis=0)
    
    # Calculate jerk
    jerks = np.diff(accelerations, axis=0)
    
    # Calculate average velocity, acceleration, and jerk
    avg_velocity = np.mean(velocities, axis=0)
    avg_acceleration = np.mean(accelerations, axis=0)
    avg_jerk = np.mean(jerks, axis=0)
    
    # Predict the next point
    last_position = positions[-1]
    next_velocity = avg_velocity + avg_acceleration + avg_jerk
    next_position = last_position - next_velocity
    
    return tuple(next_position)

def predictor_delta(points, model):
    if len(points) < 20:
        return None
    
    input_data = np.array(points[-20:]).flatten()
    input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
    prediction = predict(model, input_tensor)
    
    return tuple(prediction[0].tolist())

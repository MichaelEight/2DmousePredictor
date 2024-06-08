import math

PREDICTOR_COLORS = {
    "alpha": (255, 0, 0),  # Red
    "beta": (0, 255, 0),   # Green
    # Add more colors for additional predictors here
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

# Add additional predictor functions here

def predict_point(points):
    if len(points) < 2:
        return None
    
    dx = points[-1][0] - points[-2][0]
    dy = points[-1][1] - points[-2][1]
    
    next_x = points[-1][0] + dx
    next_y = points[-1][1] + dy
    
    return (next_x, next_y)
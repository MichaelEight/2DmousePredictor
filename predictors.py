PREDICTOR_COLORS = {
    "alpha": (255, 0, 0),
    "beta": (0, 255, 0)
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
    next_x = points[-1][0] + (dx1 + dx2) // 2
    next_y = points[-1][1] + (dy1 + dy2) // 2
    return (next_x, next_y)

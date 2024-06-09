import os
import pygame
import math
from components.config import ERROR_LIMIT, RECORDED_POSITIONS_LIMIT, CONTINUOUS_DETECTION, NUMBER_OF_PREDICTIONS

def calculate_errors(predicted, actual):
    if predicted is None or actual is None:
        return None, None
    error1 = math.sqrt((predicted[0] - actual[0]) ** 2 + (predicted[1] - actual[1]) ** 2)
    error2 = abs(predicted[0] - actual[0]) + abs(predicted[1] - actual[1])
    return error1, error2

def update_simulation(recorded_positions, last_predicted_points, predictors, past_predictions, update_counters, space_bar_pressed, mouse_positions_file):
    mouse_pos = pygame.mouse.get_pos()
    mouse_positions_file.write(f"{mouse_pos[0]}, {mouse_pos[1]}\n")
    has_mouse_moved = mouse_pos != recorded_positions[-1] if recorded_positions else False

    if space_bar_pressed and (CONTINUOUS_DETECTION or has_mouse_moved):
        recorded_positions.append(mouse_pos)
    else:
        if not recorded_positions:
            recorded_positions.append(mouse_pos)

    if len(recorded_positions) > RECORDED_POSITIONS_LIMIT:
        recorded_positions.pop(0)

    for name, predictor in predictors.items():
        points = recorded_positions[:]
        predicted_points = []

        for _ in range(NUMBER_OF_PREDICTIONS):
            predicted_point = predictor["function"](points)
            if predicted_point:
                if len(points) >= RECORDED_POSITIONS_LIMIT:
                    points.pop(0)
                points.append(predicted_point)
                predicted_points.append(predicted_point)

        if space_bar_pressed and (CONTINUOUS_DETECTION or has_mouse_moved):
            update_counters[name] += 1
            if update_counters[name] >= NUMBER_OF_PREDICTIONS:
                past_predictions[name] = [predicted_points]  # Keep only the latest past predictions
                update_counters[name] = 0

        if space_bar_pressed and (CONTINUOUS_DETECTION or has_mouse_moved) and last_predicted_points.get(name) is not None:
            error1, error2 = calculate_errors(last_predicted_points[name], mouse_pos)
            if error1 is not None and error2 is not None:
                predictor["errors"].append((error1, error2))
                if len(predictor["errors"]) > ERROR_LIMIT:
                    predictor["errors"].pop(0)
                predictor["file"].write(f"{error1}, {error2}\n")

        if space_bar_pressed and (CONTINUOUS_DETECTION or has_mouse_moved):
            last_predicted_points[name] = predicted_point

def reset_errors(predictors):
    for name, predictor in predictors.items():
        predictor["errors"] = []
        predictor["file"].close()
        predictor["file"] = open(os.path.join("data", f"errors_{name}.txt"), "w")

def reset_all(predictors, mouse_positions_file, recorded_positions, last_predicted_points):
    global past_predictions, update_counters
    recorded_positions.clear()
    past_predictions = {name: [] for name in predictors.keys()}
    update_counters = {name: 0 for name in predictors.keys()}
    last_predicted_points.clear()
    mouse_positions_file.close()
    mouse_positions_file = open(os.path.join("data", "mouse_positions.txt"), "w")

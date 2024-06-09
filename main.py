import pygame
import math
import os
import numpy as np
import torch
from predictors import predictor_delta, PREDICTOR_COLORS, get_random_color
from ml_model import MousePredictor, load_model
from shape_classifier_model import ShapeClassifier, load_classifier, predict_shape
import argparse

# Initialize Pygame
pygame.init()

# Settings
WINDOW_SIZE = (1000, 1000)
DOT_COLOR = (255, 255, 255)
CENTER_POINT_COLOR = (255, 255, 255)
POSITION_POINT_COLOR = (255, 255, 255)
POSITION_POINT_LINE_COLOR = (255, 255, 255)
TEXT_PADDING = 20
FPS_MIN = 1
FPS_MAX = 60
FPS_STEP = 5
FPS_LIMIT = 30
ERROR_LIMIT = 500
NUMBER_OF_PREDICTIONS = 1
NUMBER_OF_PREDICTIONS_STEP = 5
RECORDED_POSITIONS_LIMIT = 50
RECORDED_POSITIONS_LIMIT_STEP = 5
CONTINUOUS_DETECTION = False
DRAW_CURRENT_PREDICTIONS = True
DRAW_PAST_PREDICTIONS = True
DRAW_TRAJECTORY = True
SPACE_ONLY_MOVEMENTS = False

past_predictions = {}
update_counters = {}
space_bar_pressed = False
predicted_shape = "Not loaded"

# Create data directory if not exists
if not os.path.exists("data"):
    os.makedirs("data")

# Initialize variables
WINDOW = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Center point")
recorded_positions = []
last_predicted_points = {}

# Check if the models exist in the models_to_load directory and load them
parser = argparse.ArgumentParser()
parser.add_argument('--models_path', type=str, default='models_to_load', help='Path to the directory containing models')
args = parser.parse_args()

used_colors = set()
models = {}
for file in os.listdir(args.models_path):
    if file.endswith('.pth'):
        parts = file.split('_')
        seq_length = int(parts[0][1:])  # Extract sequence length from L20
        output_size = int(parts[1])  # Extract output size
        model_type = parts[2]
        norm_flag = parts[3].split('.')[0]
        normalize = norm_flag == "N"
        model_path = os.path.join(args.models_path, file)
        model, hidden_layers = load_model(seq_length, output_size, model_path)
        models[(seq_length, output_size, model_type, norm_flag)] = model

# Update predictors to use multiple models
predictors = {}
for (seq_length, output_size, model_type, norm_flag), model in models.items():
    color_key = f'delta_{seq_length}'
    if color_key in PREDICTOR_COLORS:
        color = PREDICTOR_COLORS[color_key]
    if color in used_colors:
        color = get_random_color(used_colors)
    used_colors.add(color)
    predictor_key = f'L{seq_length}_{output_size}_{model_type}_{norm_flag}'
    normalize = norm_flag == "N"
    predictors[predictor_key] = {
        "function": lambda points, model=model, seq_length=seq_length, output_size=output_size, normalize=normalize: predictor_delta(points, model, seq_length, output_size, normalize) if model else None,
        "color": color,
        "errors": [],
        "file": open(os.path.join("data/errors/", f"errors_{predictor_key}.txt"), "w"),
        "output_size": output_size
    }
    past_predictions[predictor_key] = []
    update_counters[predictor_key] = 0
    print(f"Loaded model: Sequence Length: {seq_length}, Output Size: {output_size}, Type: {model_type}, Normalized: {normalize}, Color: {color}")

# Load the shape classifier model
sequence_length = 20  # Adjust as needed based on your classifier training
num_classes = 2  # Adjust as needed based on your classifier training

classifier_model_path = 'trained_models/classifier_20_2_64R-32R_shapes_U.pth'  # Adjust the path accordingly
try:
    classifier_model, classifier_hidden_layers, class_map = load_classifier(sequence_length, num_classes, classifier_model_path)
    classifier_loaded = True
    print("Classifier model loaded successfully.")
except FileNotFoundError:
    classifier_loaded = False
    print("Classifier model not found.")


mouse_positions_file = open(os.path.join("data", "mouse_positions.txt"), "w")
mouse_positions_counter = 0

# Modify calculate_errors function
def calculate_errors(predicted, actual):
    if predicted is None or actual is None:
        return None, None
    error1 = math.sqrt((predicted[0] - actual[0]) ** 2 + (predicted[1] - actual[1]) ** 2)
    error2 = abs(predicted[0] - actual[0]) + abs(predicted[1] - actual[1])
    return error1, error2

# Modify update_simulation function
def update_simulation():
    global recorded_positions, last_predicted_points, predictors, past_predictions, update_counters, mouse_positions_counter

    mouse_pos = pygame.mouse.get_pos()
    has_mouse_moved = mouse_pos != recorded_positions[-1] if recorded_positions else False
    
    if (CONTINUOUS_DETECTION or has_mouse_moved) and space_bar_pressed:
        mouse_positions_file.write(f"{mouse_pos[0]}, {mouse_pos[1]}\n")
        mouse_positions_counter += 1

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

        for _ in range(NUMBER_OF_PREDICTIONS // predictor["output_size"]):
            predicted_point = predictor["function"](points)
            if predicted_point:
                points.extend(predicted_point)
                predicted_points.extend(predicted_point)

        if space_bar_pressed and (CONTINUOUS_DETECTION or has_mouse_moved):
            update_counters[name] += 1
            if update_counters[name] >= NUMBER_OF_PREDICTIONS:
                past_predictions[name] = [predicted_points]  # Keep only the latest past predictions
                update_counters[name] = 0

        if space_bar_pressed and (CONTINUOUS_DETECTION or has_mouse_moved) and last_predicted_points.get(name) is not None:
            num_comparisons = min(len(last_predicted_points[name]), len(mouse_pos))
            for i in range(num_comparisons):
                if i < len(last_predicted_points[name]) and len(mouse_pos) > 1:
                    error1, error2 = calculate_errors(last_predicted_points[name][i], mouse_pos)
                    if error1 is not None and error2 is not None:
                        predictor["errors"].append((error1, error2))
                        if len(predictor["errors"]) > ERROR_LIMIT:
                            predictor["errors"].pop(0)
                        predictor["file"].write(f"{error1}, {error2}\n")

        if space_bar_pressed and (CONTINUOUS_DETECTION or has_mouse_moved):
            last_predicted_points[name] = predicted_points


def draw_trajectory(points, color):
    for i in range(1, len(points)):
        if points[i-1] is not None and points[i] is not None:
            pygame.draw.line(WINDOW, color, points[i-1], points[i], 2)

def draw_graphics():
    WINDOW.fill((0, 0, 0))
    pygame.draw.circle(WINDOW, DOT_COLOR, (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2), 5)

    for i in range(1, len(recorded_positions)):
        pygame.draw.line(WINDOW, POSITION_POINT_LINE_COLOR, recorded_positions[i-1], recorded_positions[i], 2)
    for pos in recorded_positions:
        pygame.draw.circle(WINDOW, POSITION_POINT_COLOR, pos, 5)

    if DRAW_TRAJECTORY and DRAW_PAST_PREDICTIONS:
        for name, prediction_set in past_predictions.items():
            color = predictors[name]["color"]
            faded_color = (color[0] // 2, color[1] // 2, color[2] // 2)  # 50% opacity
            for predictions in prediction_set:
                draw_trajectory(predictions, faded_color)

    for name, predictor in predictors.items():
        if DRAW_CURRENT_PREDICTIONS:
            points = recorded_positions[:]
            predicted_points = []
            for _ in range(NUMBER_OF_PREDICTIONS // predictor["output_size"]):
                predicted_point = predictor["function"](points)
                if predicted_point is not None:
                    predicted_points.extend(predicted_point)
                    points.extend(predicted_point)
            if predicted_points and DRAW_TRAJECTORY:
                draw_trajectory(predicted_points, predictor["color"])

    render_text()
    pygame.display.update()

def render_text():
    font = pygame.font.Font(None, 36)
    y_offset = TEXT_PADDING

    # Display classifier status
    classifier_status = f"Classifier: {predicted_shape}"
    classifier_surface = font.render(classifier_status, True, (255, 255, 255))
    WINDOW.blit(classifier_surface, (TEXT_PADDING, y_offset))
    y_offset += TEXT_PADDING + classifier_surface.get_height()

    for name, predictor in predictors.items():
        avg_error = sum(e[0] for e in predictor["errors"]) / len(predictor["errors"]) if predictor["errors"] else 0
        text_surface = font.render(f"{name}: {avg_error:.2f}", True, predictor["color"])
        WINDOW.blit(text_surface, (TEXT_PADDING, y_offset))
        y_offset += TEXT_PADDING + text_surface.get_height()


def handle_events():
    global CONTINUOUS_DETECTION, FPS_LIMIT, RECORDED_POSITIONS_LIMIT, NUMBER_OF_PREDICTIONS, DRAW_PAST_PREDICTIONS, DRAW_CURRENT_PREDICTIONS, space_bar_pressed, DRAW_TRAJECTORY, SPACE_ONLY_MOVEMENTS

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            for predictor in predictors.values():
                predictor["file"].close()
            mouse_positions_file.close()
            pygame.quit()
            quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                for predictor in predictors.values():
                    predictor["file"].close()
                mouse_positions_file.close()
                pygame.quit()
                quit()
            elif event.key == pygame.K_c:
                CONTINUOUS_DETECTION = not CONTINUOUS_DETECTION
            elif event.key == pygame.K_p:
                DRAW_PAST_PREDICTIONS = not DRAW_PAST_PREDICTIONS
            elif event.key == pygame.K_o:
                DRAW_CURRENT_PREDICTIONS = not DRAW_CURRENT_PREDICTIONS
            elif event.key == pygame.K_t:
                DRAW_TRAJECTORY = not DRAW_TRAJECTORY  # Toggle trajectory
            elif event.key == pygame.K_PERIOD:
                FPS_LIMIT = min(FPS_LIMIT + FPS_STEP, FPS_MAX)
            elif event.key == pygame.K_COMMA:
                FPS_LIMIT = max(FPS_LIMIT - FPS_STEP, FPS_MIN)
            elif event.key == pygame.K_m:
                RECORDED_POSITIONS_LIMIT = min(RECORDED_POSITIONS_LIMIT + RECORDED_POSITIONS_LIMIT_STEP, 100)
            elif event.key == pygame.K_n:
                RECORDED_POSITIONS_LIMIT = max(RECORDED_POSITIONS_LIMIT - RECORDED_POSITIONS_LIMIT_STEP, RECORDED_POSITIONS_LIMIT_STEP)
            elif event.key == pygame.K_l:
                NUMBER_OF_PREDICTIONS += NUMBER_OF_PREDICTIONS_STEP
            elif event.key == pygame.K_k:
                NUMBER_OF_PREDICTIONS = max(1, NUMBER_OF_PREDICTIONS - NUMBER_OF_PREDICTIONS_STEP)
            elif event.key == pygame.K_SPACE:
                space_bar_pressed = True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE and SPACE_ONLY_MOVEMENTS:
                space_bar_pressed = False

def update_caption():
    caption_parts = [
        f"FPS: {FPS_LIMIT}",
        f"Cont: {CONTINUOUS_DETECTION}",
        f"Pts Limit: {RECORDED_POSITIONS_LIMIT}",
        f"Predictions: {NUMBER_OF_PREDICTIONS}",
        f"Draw Past: {DRAW_PAST_PREDICTIONS}",
        f"Mouse pos counter: {mouse_positions_counter}"
    ]
    pygame.display.set_caption(" | ".join(filter(None, caption_parts)))

def main():
    global CONTINUOUS_DETECTION, FPS_LIMIT, RECORDED_POSITIONS_LIMIT, last_predicted_points, space_bar_pressed, SPACE_ONLY_MOVEMENTS

    clock = pygame.time.Clock()
    last_predicted_points = {}

    if SPACE_ONLY_MOVEMENTS:
        space_bar_pressed = False
    else:
        space_bar_pressed = True

    # Write settings to settings file
    with open(os.path.join("data", "settings.txt"), "w") as settings_file:
        settings_str = f"WINDOW_SIZE: {WINDOW_SIZE}, RECORDED_POSITIONS_LIMIT: {RECORDED_POSITIONS_LIMIT}, FPS_LIMIT: {FPS_LIMIT}, CONTINUOUS_DETECTION: {CONTINUOUS_DETECTION}, NUMBER_OF_PREDICTIONS: {NUMBER_OF_PREDICTIONS}"
        settings_file.write(settings_str + "\n")
        for name, predictor in predictors.items():
            settings_file.write(f"{name}: {predictor['color']}\n")

    while True:
        handle_events()
        update_caption()
        update_simulation()
        draw_graphics()
        clock.tick(FPS_LIMIT)

if __name__ == "__main__":
    main()

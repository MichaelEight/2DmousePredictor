import pygame
import math
import os
import numpy as np
import torch
from predictors import predictor_delta, PREDICTOR_COLORS, get_random_color
from model_predictor import MousePredictor, load_model
from model_shape_classifier import ShapeClassifier, load_classifier, predict_shape
import argparse
import validate_folders_scheme as vfs
from validate_folders_scheme import folders as vfs_folders

vfs.ensure_folders_exist(vfs_folders)

def append_mouse_position(position):
    with open('data/data_mouse/mouse_positions.txt', 'a') as file:
        file.write(f"{position[0]},{position[1]}\n")

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
class_map = {}

# Initialize variables
WINDOW = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Center point")
recorded_positions = []
last_predicted_points = {}

# Check if the models exist in the models_to_load directory and load them
parser = argparse.ArgumentParser()
parser.add_argument('--models_path', type=str, default='models/models_to_load', help='Path to the directory containing models')
args = parser.parse_args()

used_colors = set()
models = {}
classifier_loaded = False

for file in os.listdir(args.models_path):
    if file.endswith('.pth'):
        parts = file.split('_')
        if file.startswith('L'):
            seq_length = int(parts[0][1:])  # Extract sequence length from L20
            output_size = int(parts[1])  # Extract output size
            model_type = parts[2]
            norm_flag = parts[3].split('.')[0]
            normalize = norm_flag == "N"
            model_path = os.path.join(args.models_path, file)
            model, hidden_layers = load_model(seq_length, output_size, model_path)
            models[(seq_length, output_size, model_type, norm_flag)] = model
        elif file.startswith('classifier'):
            sequence_length = int(parts[1])
            num_classes = int(parts[2])
            hidden_layers_str = parts[3]
            description = parts[4]
            norm_flag = parts[5].split('.')[0]
            normalize = norm_flag == "N"
            classifier_model_path = os.path.join(args.models_path, file)
            classifier_model, classifier_hidden_layers, class_map = load_classifier(sequence_length, num_classes, classifier_model_path)
            classifier_loaded = True
            print(f"Loaded Classifier: {classifier_model_path}")

if not classifier_loaded:
    print("Classifier model not found.")

# Update predictors to use multiple models
predictors = {}
for (seq_length, output_size, model_type, norm_flag), model in models.items():
    color_key = f'delta_{seq_length}'
    color = (255, 0, 0)
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
        "output_size": output_size
    }
    past_predictions[predictor_key] = []
    update_counters[predictor_key] = 0
    print(f"Loaded Predictor: Sequence Length: {seq_length}, Output Size: {output_size}, Type: {model_type}, Normalized: {normalize}, Color: {color}")

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
    global recorded_positions, last_predicted_points, predictors, past_predictions, update_counters, mouse_positions_counter, predicted_shape, probabilities

    mouse_pos = pygame.mouse.get_pos()
    has_mouse_moved = mouse_pos != recorded_positions[-1] if recorded_positions else False

    if (CONTINUOUS_DETECTION or has_mouse_moved) and space_bar_pressed:
        mouse_positions_counter += 1

    if space_bar_pressed and (CONTINUOUS_DETECTION or has_mouse_moved):
        recorded_positions.append(mouse_pos)
        append_mouse_position(mouse_pos)  # Save the new position to file
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

        if space_bar_pressed and (CONTINUOUS_DETECTION or has_mouse_moved):
            last_predicted_points[name] = predicted_points

    # Predict shape
    if classifier_loaded and len(recorded_positions) >= sequence_length:
        input_data = torch.FloatTensor([np.array(recorded_positions[-sequence_length:]).flatten()])
        predicted_class_idx, probabilities = predict_shape(classifier_model, input_data, sequence_length)
        predicted_shape = {v: k for k, v in class_map.items()}.get(predicted_class_idx, "Unknown shape")
    else:
        predicted_shape = "Not loaded"
        probabilities = []

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

def get_gradient_color(progress):
    if progress <= 0.5:
        # Interpolate between dark red and yellow
        red = int(139 + (255 - 139) * (progress / 0.5))
        green = int(0 + (255 * (progress / 0.5)))
        blue = 0
    else:
        # Interpolate between yellow and dark green
        red = int(255 - (255 - 0) * ((progress - 0.5) / 0.5))
        green = int(255 - (255 - 100) * ((progress - 0.5) / 0.5))
        blue = int(0 + (0 - 0) * ((progress - 0.5) / 0.5))
    return (red, green, blue)


def draw_progress_bar(x, y, width, height, progress):
    pygame.draw.rect(WINDOW, (0, 0, 0), (x, y, width, height))  # Background bar
    bar_color = get_gradient_color(progress)
    pygame.draw.rect(WINDOW, bar_color, (x, y, int(width * progress), height))  # Progress bar

def render_text():
    font = pygame.font.Font(None, 28)  # Font size for shapes list
    y_offset = TEXT_PADDING
    bar_width = 200  # Width of the progress bar
    bar_height = 20  # Height of the progress bar

    def draw_text_with_background(text, x, y, color, bg_color, font):
        text_surface = font.render(text, True, color)
        text_bg = pygame.Surface(text_surface.get_size())
        text_bg.fill(bg_color)
        text_bg.set_alpha(255)  # Full opacity
        text_bg.blit(text_surface, (0, 0))
        WINDOW.blit(text_bg, (x, y))

    # Display classifier status
    classifier_status = "Classifier:"
    classifier_surface = font.render(classifier_status, True, (255, 255, 255))
    classifier_bg = pygame.Surface((WINDOW_SIZE[0] - 2 * TEXT_PADDING, 30))  # Adjust size for the background
    classifier_bg.fill((0, 0, 0))
    classifier_bg.set_alpha(128)  # 50% opacity
    WINDOW.blit(classifier_bg, (TEXT_PADDING, y_offset))

    draw_text_with_background(classifier_status, TEXT_PADDING, y_offset, (255, 255, 255), (0, 0, 0), font)
    y_offset += TEXT_PADDING + 30

    if classifier_loaded and probabilities:
        shape_names = [k for k, v in sorted(class_map.items(), key=lambda item: item[1])]
        for shape_name in shape_names:
            idx = class_map[shape_name]
            prob = probabilities[idx]
            prob_text = f"{shape_name}:"

            draw_text_with_background(prob_text, TEXT_PADDING, y_offset, (255, 255, 255), (0, 0, 0), font)
            draw_progress_bar(TEXT_PADDING + 150, y_offset + 5, bar_width, bar_height, prob)
            y_offset += TEXT_PADDING + 30

    for name, predictor in predictors.items():
        avg_error = sum(e[0] for e in predictor["errors"]) / len(predictor["errors"]) if predictor["errors"] else 0
        predictor_text = f"{name}: {avg_error:.2f}"
        text_bg = pygame.Surface((WINDOW_SIZE[0] - 2 * TEXT_PADDING, 30))  # Adjust size for the background
        text_bg.fill((0, 0, 0))
        text_bg.set_alpha(128)  # 50% opacity
        WINDOW.blit(text_bg, (TEXT_PADDING, y_offset))

        draw_text_with_background(predictor_text, TEXT_PADDING, y_offset, predictor["color"], (0, 0, 0), font)
        y_offset += TEXT_PADDING + 30


def handle_events():
    global CONTINUOUS_DETECTION, FPS_LIMIT, RECORDED_POSITIONS_LIMIT, NUMBER_OF_PREDICTIONS, DRAW_PAST_PREDICTIONS, DRAW_CURRENT_PREDICTIONS, space_bar_pressed, DRAW_TRAJECTORY, SPACE_ONLY_MOVEMENTS

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
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

    while True:
        handle_events()
        update_caption()
        update_simulation()
        draw_graphics()
        clock.tick(FPS_LIMIT)

if __name__ == "__main__":
    main()

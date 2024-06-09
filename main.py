import pygame
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from predictors import predictor_alpha, predictor_beta, predictor_gamma, predictor_delta, PREDICTOR_COLORS
from ml_model import MousePredictor, train_model, save_model, load_model, predict

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
FPS_STEP = 1
FPS_LIMIT = 10
ERROR_LIMIT = 500
NUMBER_OF_PREDICTIONS = 1
RECORDED_POSITIONS_LIMIT = 50
RECORDED_POSITIONS_LIMIT_STEP = 5
CONTINUOUS_DETECTION = False
DRAW_CURRENT_PREDICTIONS = True
DRAW_PAST_PREDICTIONS = True
DRAW_TRAJECTORY = True
TRAIN_EVERY_N_UPDATES = 50

past_predictions = {name: [] for name in PREDICTOR_COLORS.keys()}
update_counters = {name: 0 for name in PREDICTOR_COLORS.keys()}
space_bar_pressed = False
train_counter = 0

# Create data directory if not exists
if not os.path.exists("data"):
    os.makedirs("data")

# Initialize variables
WINDOW = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Center point")
recorded_positions = []
last_predicted_points = {}

# Initialize the model
model = MousePredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model = load_model(model)

predictors = {
    "alpha": {
        "function": predictor_alpha,
        "color": PREDICTOR_COLORS["alpha"],
        "errors": [],
        "file": open(os.path.join("data", "errors_alpha.txt"), "w")
    },
    "beta": {
        "function": predictor_beta,
        "color": PREDICTOR_COLORS["beta"],
        "errors": [],
        "file": open(os.path.join("data", "errors_beta.txt"), "w")
    },
    "gamma": {
        "function": predictor_gamma,
        "color": PREDICTOR_COLORS["gamma"],
        "errors": [],
        "file": open(os.path.join("data", "errors_gamma.txt"), "w")
    },
    "delta": {
        "function": lambda points: predictor_delta(points, model),
        "color": PREDICTOR_COLORS["delta"],
        "errors": [],
        "file": open(os.path.join("data", "errors_delta.txt"), "w")
    }
}

mouse_positions_file = open(os.path.join("data", "mouse_positions.txt"), "w")

def calculate_errors(predicted, actual):
    if predicted is None or actual is None:
        return None, None
    error1 = math.sqrt((predicted[0] - actual[0]) ** 2 + (predicted[1] - actual[1]) ** 2)
    error2 = abs(predicted[0] - actual[0]) + abs(predicted[1] - actual[1])
    return error1, error2

def update_simulation():
    global recorded_positions, last_predicted_points, predictors, past_predictions, update_counters, train_counter, model

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

    # Train the model every TRAIN_EVERY_N_UPDATES updates
    if len(recorded_positions) >= 21 and train_counter % TRAIN_EVERY_N_UPDATES == 0:
        input_data = np.array(recorded_positions[-21:-1]).flatten()
        target_data = np.array(recorded_positions[-1])
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
        target_tensor = torch.FloatTensor(target_data).unsqueeze(0)
        loss = train_model(model, input_tensor, target_tensor, criterion, optimizer)
        print(f"Training loss: {loss}")
        save_model(model)

    train_counter += 1

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

    if DRAW_TRAJECTORY:
        for name, prediction_set in past_predictions.items():
            color = predictors[name]["color"]
            faded_color = (color[0] // 2, color[1] // 2, color[2] // 2)  # 50% opacity
            for predictions in prediction_set:
                draw_trajectory(predictions, faded_color)

    for name, predictor in predictors.items():
        if DRAW_CURRENT_PREDICTIONS:
            points = recorded_positions[:]
            predicted_points = []
            for _ in range(NUMBER_OF_PREDICTIONS):
                predicted_point = predictor["function"](points)
                if predicted_point is not None:
                    predicted_points.append(predicted_point)
                    if len(points) >= RECORDED_POSITIONS_LIMIT:
                        points.pop(0)
                    points.append(predicted_point)
            if predicted_points:
                draw_trajectory(predicted_points, predictor["color"])

    render_text()
    pygame.display.update()


def render_text():
    font = pygame.font.Font(None, 36)
    y_offset = TEXT_PADDING

    for name, predictor in predictors.items():
        avg_error = sum(e[0] for e in predictor["errors"]) / len(predictor["errors"]) if predictor["errors"] else 0
        text_surface = font.render(f"{name}: {avg_error:.2f}", True, predictor["color"])
        WINDOW.blit(text_surface, (TEXT_PADDING, y_offset))
        y_offset += TEXT_PADDING + text_surface.get_height()

def handle_events():
    global CONTINUOUS_DETECTION, FPS_LIMIT, RECORDED_POSITIONS_LIMIT, NUMBER_OF_PREDICTIONS, DRAW_PAST_PREDICTIONS, DRAW_CURRENT_PREDICTIONS, space_bar_pressed, DRAW_TRAJECTORY

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
            elif event.key == pygame.K_s:
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
                NUMBER_OF_PREDICTIONS += 1
            elif event.key == pygame.K_k:
                NUMBER_OF_PREDICTIONS = max(1, NUMBER_OF_PREDICTIONS - 1)
            elif event.key == pygame.K_SPACE:
                space_bar_pressed = True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                space_bar_pressed = False

def update_caption():
    caption_parts = [
        f"FPS: {FPS_LIMIT}",
        f"Cont: {CONTINUOUS_DETECTION}",
        f"Pts Limit: {RECORDED_POSITIONS_LIMIT}",
        f"Predictions: {NUMBER_OF_PREDICTIONS}",
        f"Draw Past: {DRAW_PAST_PREDICTIONS}"
    ]
    pygame.display.set_caption(" | ".join(filter(None, caption_parts)))

def main():
    global CONTINUOUS_DETECTION, FPS_LIMIT, RECORDED_POSITIONS_LIMIT, last_predicted_points, space_bar_pressed

    clock = pygame.time.Clock()
    last_predicted_points = {}
    space_bar_pressed = False

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

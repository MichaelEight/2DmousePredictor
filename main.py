import pygame
import math
import os
from predictors import predictor_alpha, predictor_beta, PREDICTOR_COLORS

# Initialize Pygame
pygame.init()

# Settings
WINDOW_SIZE = (1000, 1000)
DOT_COLOR = (255, 255, 255)
CENTER_POINT_COLOR = (255, 255, 255)
POSITION_POINT_COLOR = (255, 255, 255)
POSITION_POINT_LINE_COLOR = (255, 255, 255)
TEXT_PADDING = 20
MIN_FPS = 1
MAX_FPS = 120
ERROR_LIMIT = 500
RECORDED_POSITIONS_LIMIT = 50
FPS_LIMIT = 10
CONTINUOUS_DETECTION = False
NUMBER_OF_PREDICTIONS = 5
DRAW_CURRENT_PREDICTIONS = False
DRAW_PAST_PREDICTIONS = True

past_predictions = {name: [] for name in PREDICTOR_COLORS.keys()}
update_counters = {name: 0 for name in PREDICTOR_COLORS.keys()}

# Create data directory if not exists
if not os.path.exists("data"):
    os.makedirs("data")

# Initialize variables
WINDOW = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Center point")
recorded_positions = []
past_predictions = {name: [] for name in PREDICTOR_COLORS.keys()}
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
    }
    # Add additional predictors here
}

mouse_positions_file = open(os.path.join("data", "mouse_positions.txt"), "w")

def calculate_errors(predicted, actual):
    if predicted is None or actual is None:
        return None, None
    error1 = math.sqrt((predicted[0] - actual[0]) ** 2 + (predicted[1] - actual[1]) ** 2)
    error2 = abs(predicted[0] - actual[0]) + abs(predicted[1] - actual[1])
    return error1, error2

def render():
    global recorded_positions, last_predicted_points, predictors, past_predictions, update_counters

    mouse_pos = pygame.mouse.get_pos()
    mouse_positions_file.write(f"{mouse_pos[0]}, {mouse_pos[1]}\n")
    has_mouse_moved = mouse_pos != recorded_positions[-1] if recorded_positions else False

    if CONTINUOUS_DETECTION or has_mouse_moved:
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
                if DRAW_CURRENT_PREDICTIONS:
                    pygame.draw.circle(WINDOW, predictor["color"], predicted_point, 5)

        if (CONTINUOUS_DETECTION or has_mouse_moved):
            update_counters[name] += 1
            if len(past_predictions[name]) > NUMBER_OF_PREDICTIONS:
                        past_predictions[name] = past_predictions[name][-NUMBER_OF_PREDICTIONS:]
            if update_counters[name] >= NUMBER_OF_PREDICTIONS:
                if DRAW_PAST_PREDICTIONS:
                    past_predictions[name].append(predicted_points)
                update_counters[name] = 0

        if (CONTINUOUS_DETECTION or has_mouse_moved) and last_predicted_points.get(name) is not None:
            error1, error2 = calculate_errors(last_predicted_points[name], mouse_pos)
            if error1 is not None and error2 is not None:
                predictor["errors"].append((error1, error2))
                if len(predictor["errors"]) > ERROR_LIMIT:
                    predictor["errors"].pop(0)
                predictor["file"].write(f"{error1}, {error2}\n")

        if CONTINUOUS_DETECTION or has_mouse_moved:
            last_predicted_points[name] = predicted_point

    for i in range(1, len(recorded_positions)):
        pygame.draw.line(WINDOW, POSITION_POINT_LINE_COLOR, recorded_positions[i-1], recorded_positions[i], 2)
    for pos in recorded_positions:
        pygame.draw.circle(WINDOW, POSITION_POINT_COLOR, pos, 5)

    if DRAW_PAST_PREDICTIONS:
        render_past_predictions()
    render_text()

def render_past_predictions():
    for name, prediction_set in past_predictions.items():
        color = predictors[name]["color"]
        faded_color = (color[0] // 2, color[1] // 2, color[2] // 2)  # 50% opacity
        for predictions in prediction_set:
            for point in predictions:
                pygame.draw.circle(WINDOW, faded_color, point, 5)

def render_text():
    font = pygame.font.Font(None, 36)
    y_offset = TEXT_PADDING

    for name, predictor in predictors.items():
        avg_error = sum(e[0] for e in predictor["errors"]) / len(predictor["errors"]) if predictor["errors"] else 0
        text_surface = font.render(f"{name}: {avg_error:.2f}", True, predictor["color"])
        WINDOW.blit(text_surface, (TEXT_PADDING, y_offset))
        y_offset += TEXT_PADDING + text_surface.get_height()

def handle_events():
    global CONTINUOUS_DETECTION, FPS_LIMIT, RECORDED_POSITIONS_LIMIT, NUMBER_OF_PREDICTIONS, DRAW_PAST_PREDICTIONS, DRAW_CURRENT_PREDICTIONS

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
            elif event.key == pygame.K_PERIOD:
                FPS_LIMIT = min(FPS_LIMIT + 5, MAX_FPS)
            elif event.key == pygame.K_COMMA:
                FPS_LIMIT = max(FPS_LIMIT - 5, MIN_FPS)
            elif event.key == pygame.K_m:
                RECORDED_POSITIONS_LIMIT = min(RECORDED_POSITIONS_LIMIT + 5, 100)
            elif event.key == pygame.K_n:
                RECORDED_POSITIONS_LIMIT = max(RECORDED_POSITIONS_LIMIT - 5, 5)
            elif event.key == pygame.K_l:
                NUMBER_OF_PREDICTIONS += 1
            elif event.key == pygame.K_k:
                NUMBER_OF_PREDICTIONS = max(1, NUMBER_OF_PREDICTIONS - 1)

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
    global CONTINUOUS_DETECTION, FPS_LIMIT, RECORDED_POSITIONS_LIMIT, last_predicted_points

    clock = pygame.time.Clock()
    last_predicted_points = {}

    # Write settings to settings file
    with open(os.path.join("data", "settings.txt"), "w") as settings_file:
        settings_str = f"WINDOW_SIZE: {WINDOW_SIZE}, RECORDED_POSITIONS_LIMIT: {RECORDED_POSITIONS_LIMIT}, FPS_LIMIT: {FPS_LIMIT}, CONTINUOUS_DETECTION: {CONTINUOUS_DETECTION}, NUMBER_OF_PREDICTIONS: {NUMBER_OF_PREDICTIONS}"
        settings_file.write(settings_str + "\n")
        for name, predictor in predictors.items():
            settings_file.write(f"{name}: {predictor['color']}\n")

    while True:
        handle_events()
        update_caption()

        WINDOW.fill((0, 0, 0))
        pygame.draw.circle(WINDOW, DOT_COLOR, (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2), 5)
        render()
        pygame.display.update()
        clock.tick(FPS_LIMIT)

if __name__ == "__main__":
    main()

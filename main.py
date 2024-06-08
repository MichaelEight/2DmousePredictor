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
CONTINUOUS_DETECTION = True

# Create data directory if not exists
if not os.path.exists("data"):
    os.makedirs("data")

# Initialize variables
WINDOW = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Center point")
recorded_positions = []
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

def calculate_errors(predicted, actual):
    if predicted is None or actual is None:
        return None, None
    error1 = math.sqrt((predicted[0] - actual[0]) ** 2 + (predicted[1] - actual[1]) ** 2)
    error2 = abs(predicted[0] - actual[0]) + abs(predicted[1] - actual[1])
    return error1, error2

def render():
    global recorded_positions, last_predicted_points, predictors

    mouse_pos = pygame.mouse.get_pos()
    has_mouse_moved = mouse_pos != recorded_positions[-1] if recorded_positions else False

    if CONTINUOUS_DETECTION:
        recorded_positions.append(mouse_pos)
    else:
        if not recorded_positions or has_mouse_moved:
            recorded_positions.append(mouse_pos)

    if len(recorded_positions) > RECORDED_POSITIONS_LIMIT:
        recorded_positions.pop(0)

    for name, predictor in predictors.items():
        predicted_point = predictor["function"](recorded_positions)
        
        if (CONTINUOUS_DETECTION or has_mouse_moved) and last_predicted_points.get(name) is not None:
            error1, error2 = calculate_errors(last_predicted_points[name], mouse_pos)
            if error1 is not None and error2 is not None:
                predictor["errors"].append((error1, error2))
                if len(predictor["errors"]) > ERROR_LIMIT:
                    predictor["errors"].pop(0)
                predictor["file"].write(f"{error1}, {error2}\n")
        
        if CONTINUOUS_DETECTION or has_mouse_moved:
            last_predicted_points[name] = predicted_point

        if predicted_point:
            pygame.draw.circle(WINDOW, predictor["color"], predicted_point, 5)

    for i in range(1, len(recorded_positions)):
        pygame.draw.line(WINDOW, POSITION_POINT_LINE_COLOR, recorded_positions[i-1], recorded_positions[i], 2)
    for pos in recorded_positions:
        pygame.draw.circle(WINDOW, POSITION_POINT_COLOR, pos, 5)

    render_text()

def render_text():
    font = pygame.font.Font(None, 36)
    y_offset = TEXT_PADDING

    for name, predictor in predictors.items():
        avg_error = sum(e[0] for e in predictor["errors"]) / len(predictor["errors"]) if predictor["errors"] else 0
        text_surface = font.render(f"{name}: {avg_error:.2f}", True, predictor["color"])
        WINDOW.blit(text_surface, (TEXT_PADDING, y_offset))
        y_offset += TEXT_PADDING + text_surface.get_height()

def handle_events():
    global CONTINUOUS_DETECTION, FPS_LIMIT, RECORDED_POSITIONS_LIMIT

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            for predictor in predictors.values():
                predictor["file"].close()
            pygame.quit()
            quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                for predictor in predictors.values():
                    predictor["file"].close()
                pygame.quit()
                quit()
            elif event.key == pygame.K_s:
                CONTINUOUS_DETECTION = not CONTINUOUS_DETECTION
            elif event.key == pygame.K_PERIOD:
                FPS_LIMIT = min(FPS_LIMIT + 5, MAX_FPS)
            elif event.key == pygame.K_COMMA:
                FPS_LIMIT = max(FPS_LIMIT - 5, MIN_FPS)
            elif event.key == pygame.K_m:
                RECORDED_POSITIONS_LIMIT = min(RECORDED_POSITIONS_LIMIT + 5, 100)
            elif event.key == pygame.K_n:
                RECORDED_POSITIONS_LIMIT = max(RECORDED_POSITIONS_LIMIT - 5, 5)

def main():
    global CONTINUOUS_DETECTION, FPS_LIMIT, RECORDED_POSITIONS_LIMIT, last_predicted_points

    clock = pygame.time.Clock()
    last_predicted_points = {}

    # Write settings to error files
    settings_str = f"WINDOW_SIZE: {WINDOW_SIZE}, RECORDED_POSITIONS_LIMIT: {RECORDED_POSITIONS_LIMIT}, FPS_LIMIT: {FPS_LIMIT}, CONTINUOUS_DETECTION: {CONTINUOUS_DETECTION}"
    for predictor in predictors.values():
        predictor["file"].write(settings_str + f" PREDICTOR_COLOR: {predictor['color']}\n")

    while True:
        handle_events()

        WINDOW.fill((0, 0, 0))
        pygame.draw.circle(WINDOW, DOT_COLOR, (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2), 5)
        render()
        pygame.display.update()
        clock.tick(FPS_LIMIT)

if __name__ == "__main__":
    main()

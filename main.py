import pygame
import math
from predictor_alpha import predict_point as predict

# Initialize Pygame
pygame.init()

# Settings
WINDOW_SIZE = (1000, 1000)
DOT_COLOR = (255, 255, 255)
CENTER_POINT_COLOR = (255, 255, 255)
POSITION_POINT_COLOR = (255, 255, 255)
POSITION_POINT_LINE_COLOR = (255, 255, 255)
POSITION_POINT_PREDICTED_COLOR = (255, 0, 0)
MIN_FPS = 1
MAX_FPS = 120
ERROR_LIMIT = 500
RECORDED_POSITIONS_LIMIT = 10
FPS_LIMIT = 30
CONTINUOUS_DETECTION = True

# Initialize variables
WINDOW = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Center point")
recorded_positions = []
errors_euclidean = []
errors_manhattan = []
predicted_point = None
last_predicted_point = None

def calculate_errors(predicted, actual):
    if predicted is None or actual is None:
        return None, None
    error1 = math.sqrt((predicted[0] - actual[0]) ** 2 + (predicted[1] - actual[1]) ** 2)
    error2 = abs(predicted[0] - actual[0]) + abs(predicted[1] - actual[1])
    return error1, error2

def render():
    global recorded_positions, last_predicted_point, errors_euclidean, errors_manhattan, predicted_point

    mouse_pos = pygame.mouse.get_pos()
    has_mouse_moved = mouse_pos != recorded_positions[-1] if recorded_positions else False

    if CONTINUOUS_DETECTION:
        recorded_positions.append(mouse_pos)
    else:
        if not recorded_positions or has_mouse_moved:
            recorded_positions.append(mouse_pos)

    if len(recorded_positions) > RECORDED_POSITIONS_LIMIT:
        recorded_positions.pop(0)

    if CONTINUOUS_DETECTION or has_mouse_moved:
        predicted_point = predict(recorded_positions)
    
    if (CONTINUOUS_DETECTION or has_mouse_moved) and last_predicted_point is not None:
        error1, error2 = calculate_errors(last_predicted_point, mouse_pos)
        if error1 is not None and error2 is not None:
            errors_euclidean.append(error1)
            errors_manhattan.append(error2)
            if len(errors_euclidean) > ERROR_LIMIT:
                errors_euclidean.pop(0)
            if len(errors_manhattan) > ERROR_LIMIT:
                errors_manhattan.pop(0)

    if CONTINUOUS_DETECTION or has_mouse_moved:
        last_predicted_point = predicted_point

    for i in range(1, len(recorded_positions)):
        pygame.draw.line(WINDOW, POSITION_POINT_LINE_COLOR, recorded_positions[i-1], recorded_positions[i], 2)
    for pos in recorded_positions:
        pygame.draw.circle(WINDOW, POSITION_POINT_COLOR, pos, 5)

    if predicted_point:
        pygame.draw.circle(WINDOW, POSITION_POINT_PREDICTED_COLOR, predicted_point, 5)

def update_caption():
    avg_error1 = sum(errors_euclidean) / len(errors_euclidean) if errors_euclidean else 0
    avg_error2 = sum(errors_manhattan) / len(errors_manhattan) if errors_manhattan else 0
    caption_parts = [
        f"FPS: {FPS_LIMIT}",
        f"Cont: {CONTINUOUS_DETECTION}",
        f"Pts Limit: {RECORDED_POSITIONS_LIMIT}",
        f"Avg Error (Euc): {avg_error1:.2f}",
        f"Avg Error (Man): {avg_error2:.2f}",
        f"Cur Error (Euc): {errors_euclidean[-1]:.2f}" if errors_euclidean else "",
        f"Cur Error (Man): {errors_manhattan[-1]:.2f}" if errors_manhattan else "",
    ]
    pygame.display.set_caption(" | ".join(filter(None, caption_parts)))

def main():
    global CONTINUOUS_DETECTION, FPS_LIMIT, RECORDED_POSITIONS_LIMIT

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
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

        update_caption()

        WINDOW.fill((0, 0, 0))
        pygame.draw.circle(WINDOW, DOT_COLOR, (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2), 5)
        render()
        pygame.display.update()
        clock.tick(FPS_LIMIT)

if __name__ == "__main__":
    main()

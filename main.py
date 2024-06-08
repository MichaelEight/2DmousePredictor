import pygame
import math
from predictor_alpha import predict_point as predict

# Initialize Pygame
pygame.init()

# Set the window size
WINDOW_SIZE = (1000, 1000)
WINDOW = pygame.display.set_mode(WINDOW_SIZE)

# Set the title of the window
pygame.display.set_caption("Center point")

# Set the color of the dot
DOT_COLOR = (255, 255, 255)

# Set the coordinates of the center point
CENTER_POINT = (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2)

# CONSTS
recorded_positions = []
POSITION_POINT_COLOR = (255, 255, 255)
POSITION_POINT_LINE_COLOR = (255, 255, 255)
POSITION_POINT_PREDICTED_COLOR = (255, 0, 0)
MIN_FPS = 1
MAX_FPS = 120
ERROR_LIMIT = 500

# SETTINGS
RECORDED_POSITIONS_LIMIT = 10
continuous_detection = True
fps_limit = 30

# Variables to store errors
errors_euclidean = []
errors_manhattan = []
predicted_point = None
last_predicted_point = None

def calculate_errors(predicted, actual):
    if predicted is None or actual is None:
        return None, None
    # Euclidean distance
    error1 = math.sqrt((predicted[0] - actual[0]) ** 2 + (predicted[1] - actual[1]) ** 2)
    # Manhattan distance
    error2 = abs(predicted[0] - actual[0]) + abs(predicted[1] - actual[1])
    return error1, error2

def render():
    global recorded_positions, last_predicted_point, errors_euclidean, errors_manhattan, predicted_point

    # If the mouse has moved or continuous_detection is True, get its position
    mouse_pos = pygame.mouse.get_pos()

    has_mouse_moved = mouse_pos != recorded_positions[-1] if len(recorded_positions) > 0 else False

    if continuous_detection:
        recorded_positions.append(mouse_pos)
    else:
        if not recorded_positions or has_mouse_moved:
            recorded_positions.append(mouse_pos)

    # If the array exceeds the recorded positions limit, remove the oldest element
    if len(recorded_positions) > RECORDED_POSITIONS_LIMIT:
        recorded_positions.pop(0)

    # Predict the next point and draw it
    if continuous_detection or has_mouse_moved:
        predicted_point = predict(recorded_positions)
    
    # Calculate and store errors
    if (continuous_detection or has_mouse_moved) and last_predicted_point is not None:
        error1, error2 = calculate_errors(last_predicted_point, mouse_pos)
        if error1 is not None and error2 is not None:
            errors_euclidean.append(error1)
            errors_manhattan.append(error2)
            
            if len(errors_euclidean) > ERROR_LIMIT:
                errors_euclidean.pop(0)
            if len(errors_manhattan) > ERROR_LIMIT:
                errors_manhattan.pop(0)
    
    # Update last predicted point
    if continuous_detection or has_mouse_moved:
        last_predicted_point = predicted_point

    # Draw all points in the array as a white line
    for i in range(1, len(recorded_positions)):
        pygame.draw.line(WINDOW, POSITION_POINT_LINE_COLOR, recorded_positions[i-1], recorded_positions[i], 2)
    for pos in recorded_positions:
        pygame.draw.circle(WINDOW, POSITION_POINT_COLOR, pos, 5)

    if predicted_point:
        pygame.draw.circle(WINDOW, POSITION_POINT_PREDICTED_COLOR, predicted_point, 5)

clock = pygame.time.Clock()

while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                quit()

            elif event.key == pygame.K_s:
                continuous_detection = not continuous_detection

            # FPS_LIMIT CONTROL
            elif event.key == pygame.K_PERIOD:
                if fps_limit < MAX_FPS:
                    fps_limit += 5
                    if fps_limit > MAX_FPS:
                        fps_limit = MAX_FPS
            elif event.key == pygame.K_COMMA:
                if fps_limit > MIN_FPS:
                    fps_limit -= 5
                    if fps_limit < MIN_FPS:
                        fps_limit = MIN_FPS
                elif fps_limit == MIN_FPS: # Revert to 5-based ending
                    fps_limit = 5

            # RECORDED_POSITIONS_LIMIT CONTROL
            elif event.key == pygame.K_m:
                RECORDED_POSITIONS_LIMIT += 5
                if RECORDED_POSITIONS_LIMIT > 100:
                    RECORDED_POSITIONS_LIMIT = 100
            elif event.key == pygame.K_n:
                if RECORDED_POSITIONS_LIMIT > 5:
                    RECORDED_POSITIONS_LIMIT -= 5
                    if RECORDED_POSITIONS_LIMIT < 5:
                        RECORDED_POSITIONS_LIMIT = 5

    # Calculate average errors
    avg_error1 = sum(errors_euclidean) / len(errors_euclidean) if errors_euclidean else 0
    avg_error2 = sum(errors_manhattan) / len(errors_manhattan) if errors_manhattan else 0

    # Set the title of the window
    caption_parts = [
        f"FPS: {fps_limit}",
        f"Cont: {continuous_detection}",
        f"Pts Limit: {RECORDED_POSITIONS_LIMIT}",
        f"Avg Error (Euc): {avg_error1:.2f}",
        f"Avg Error (Man): {avg_error2:.2f}",
        f"Cur Error (Euc): {errors_euclidean[-1]:.2f}" if errors_euclidean else "",
        f"Cur Error (Man): {errors_manhattan[-1]:.2f}" if errors_manhattan else "",
    ]
    pygame.display.set_caption(" | ".join(filter(None, caption_parts)))

    # Clear the window
    WINDOW.fill((0, 0, 0))

    # Draw the dot
    pygame.draw.circle(WINDOW, DOT_COLOR, CENTER_POINT, 5)

    render()

    # Update the window
    pygame.display.update()

    # Limit FPS
    clock.tick(fps_limit)

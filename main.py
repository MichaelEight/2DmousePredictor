import pygame

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

def coords(x, y):
    return (x - CENTER_POINT[0], y - CENTER_POINT[1])

RECORDED_POSITIONS_LIMIT = 10
POSITION_POINT_COLOR = (255, 255, 255)
recorded_positions = []

continuous_detection = False

fps_limit = 30
MIN_FPS = 1
MAX_FPS = 120

def render():
    global recorded_positions

    # If the mouse has moved or continuous_detection is True, get its position
    mouse_pos = pygame.mouse.get_pos()

    if continuous_detection:
        recorded_positions.append(mouse_pos)
    else:
        if not recorded_positions or mouse_pos != recorded_positions[-1]:
            recorded_positions.append(mouse_pos)

    # If the array exceeds the recorded positions limit, remove the oldest element
    if len(recorded_positions) > RECORDED_POSITIONS_LIMIT:
        recorded_positions.pop(0)

    # Draw all points in the array as a white line
    for i in range(1, len(recorded_positions)):
        pygame.draw.line(WINDOW, (255, 255, 255), recorded_positions[i-1], recorded_positions[i], 2)
    for pos in recorded_positions:
        pygame.draw.circle(WINDOW, POSITION_POINT_COLOR, pos, 5)


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
                    

    # Set the title of the window
    pygame.display.set_caption("FPS Limit: " + str(fps_limit)
                               + " | Continuous: " + str(continuous_detection)
                               + " | Recorded Positions Limit: " + str(RECORDED_POSITIONS_LIMIT))

    # Clear the window
    WINDOW.fill((0, 0, 0))

    # Draw the dot
    pygame.draw.circle(WINDOW, DOT_COLOR, CENTER_POINT, 5)

    render()

    # Update the window
    pygame.display.update()

    # Limit FPS to 30
    clock.tick(fps_limit)

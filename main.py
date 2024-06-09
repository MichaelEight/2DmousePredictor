import pygame
import os
from components.config import (
    WINDOW_SIZE, FPS_MIN, FPS_MAX, FPS_STEP, ERROR_LIMIT, RECORDED_POSITIONS_LIMIT, FPS_LIMIT, CONTINUOUS_DETECTION, NUMBER_OF_PREDICTIONS, DRAW_CURRENT_PREDICTIONS, DRAW_PAST_PREDICTIONS
)
from components.events import handle_events
from components.graphics import draw_graphics, update_caption
from components.simulation import update_simulation, reset_errors, reset_all
from components.utils import create_data_directory, initialize_predictors, initialize_files
from predictors import PREDICTOR_COLORS

# Initialize Pygame
pygame.init()

# Create data directory if not exists
create_data_directory()

# Initialize variables
WINDOW = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Center point")
recorded_positions = []
last_predicted_points = {}
past_predictions = {name: [] for name in PREDICTOR_COLORS.keys()}
update_counters = {name: 0 for name in PREDICTOR_COLORS.keys()}
space_bar_pressed = False

predictors = initialize_predictors()
mouse_positions_file = initialize_files(predictors)

def main():
    global CONTINUOUS_DETECTION, FPS_LIMIT, RECORDED_POSITIONS_LIMIT, NUMBER_OF_PREDICTIONS, DRAW_PAST_PREDICTIONS, DRAW_CURRENT_PREDICTIONS, space_bar_pressed

    clock = pygame.time.Clock()
    space_bar_pressed = False

    while True:
        CONTINUOUS_DETECTION, FPS_LIMIT, RECORDED_POSITIONS_LIMIT, NUMBER_OF_PREDICTIONS, DRAW_PAST_PREDICTIONS, DRAW_CURRENT_PREDICTIONS, space_bar_pressed = handle_events(
            CONTINUOUS_DETECTION, FPS_LIMIT, RECORDED_POSITIONS_LIMIT, NUMBER_OF_PREDICTIONS, DRAW_PAST_PREDICTIONS, DRAW_CURRENT_PREDICTIONS, space_bar_pressed, predictors, mouse_positions_file, recorded_positions, last_predicted_points, past_predictions)
        update_caption(WINDOW, FPS_LIMIT, CONTINUOUS_DETECTION, RECORDED_POSITIONS_LIMIT, NUMBER_OF_PREDICTIONS, DRAW_PAST_PREDICTIONS)
        update_simulation(recorded_positions, last_predicted_points, predictors, past_predictions, update_counters, space_bar_pressed, mouse_positions_file)
        draw_graphics(WINDOW, recorded_positions, predictors, last_predicted_points, DRAW_CURRENT_PREDICTIONS, DRAW_PAST_PREDICTIONS, past_predictions)
        clock.tick(FPS_LIMIT)

if __name__ == "__main__":
    main()

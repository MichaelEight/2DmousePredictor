import os
import pygame
from predictors import PREDICTOR_COLORS

def create_data_directory():
    if not os.path.exists("data"):
        os.makedirs("data")

def initialize_predictors():
    from predictors import predictor_alpha, predictor_beta
    return {
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

def initialize_files(predictors):
    mouse_positions_file = open(os.path.join("data", "mouse_positions.txt"), "w")
    return mouse_positions_file

def render_text(WINDOW, predictors):
    font = pygame.font.Font(None, 36)
    y_offset = 20

    for name, predictor in predictors.items():
        avg_error = sum(e[0] for e in predictor["errors"]) / len(predictor["errors"]) if predictor["errors"] else 0
        text_surface = font.render(f"{name}: {avg_error:.2f}", True, predictor["color"])
        WINDOW.blit(text_surface, (20, y_offset))
        y_offset += 20 + text_surface.get_height()

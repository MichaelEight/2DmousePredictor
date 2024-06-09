import pygame
from components.simulation import reset_errors, reset_all
from components.config import FPS_MIN, FPS_MAX, FPS_STEP, RECORDED_POSITIONS_LIMIT_STEP

def handle_events(CONTINUOUS_DETECTION, FPS_LIMIT, RECORDED_POSITIONS_LIMIT, NUMBER_OF_PREDICTIONS, DRAW_PAST_PREDICTIONS, DRAW_CURRENT_PREDICTIONS, space_bar_pressed, predictors, mouse_positions_file, recorded_positions, last_predicted_points, past_predictions):
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
            elif event.key == pygame.K_e:
                reset_errors(predictors)
            elif event.key == pygame.K_r:
                reset_all(predictors, mouse_positions_file, recorded_positions, last_predicted_points, past_predictions)
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                space_bar_pressed = False

    return CONTINUOUS_DETECTION, FPS_LIMIT, RECORDED_POSITIONS_LIMIT, NUMBER_OF_PREDICTIONS, DRAW_PAST_PREDICTIONS, DRAW_CURRENT_PREDICTIONS, space_bar_pressed

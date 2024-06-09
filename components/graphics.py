import pygame
from components.config import (
    DOT_COLOR, CENTER_POINT_COLOR, POSITION_POINT_COLOR, POSITION_POINT_LINE_COLOR, TEXT_PADDING, WINDOW_SIZE, NUMBER_OF_PREDICTIONS, RECORDED_POSITIONS_LIMIT
)
from components.utils import render_text

def draw_graphics(WINDOW, recorded_positions, predictors, last_predicted_points, DRAW_CURRENT_PREDICTIONS, DRAW_PAST_PREDICTIONS, past_predictions):
    WINDOW.fill((0, 0, 0))
    pygame.draw.circle(WINDOW, DOT_COLOR, (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2), 5)

    for i in range(1, len(recorded_positions)):
        pygame.draw.line(WINDOW, POSITION_POINT_LINE_COLOR, recorded_positions[i-1], recorded_positions[i], 2)
    for pos in recorded_positions:
        pygame.draw.circle(WINDOW, POSITION_POINT_COLOR, pos, 5)

    if DRAW_PAST_PREDICTIONS:
        for name, prediction_set in past_predictions.items():
            color = predictors[name]["color"]
            faded_color = (color[0] // 2, color[1] // 2, color[2] // 2)  # 50% opacity
            for predictions in prediction_set:
                for point in predictions:
                    pygame.draw.circle(WINDOW, faded_color, point, 5)

    if DRAW_CURRENT_PREDICTIONS:
        for name, predictor in predictors.items():
            points = recorded_positions[:]
            for _ in range(NUMBER_OF_PREDICTIONS):
                predicted_point = predictor["function"](points)
                if predicted_point:
                    pygame.draw.circle(WINDOW, predictor["color"], predicted_point, 5)
                    if len(points) >= RECORDED_POSITIONS_LIMIT:
                        points.pop(0)
                    points.append(predicted_point)

    render_text(WINDOW, predictors)
    pygame.display.update()

def update_caption(WINDOW, FPS_LIMIT, CONTINUOUS_DETECTION, RECORDED_POSITIONS_LIMIT, NUMBER_OF_PREDICTIONS, DRAW_PAST_PREDICTIONS):
    caption_parts = [
        f"FPS: {FPS_LIMIT}",
        f"Cont: {CONTINUOUS_DETECTION}",
        f"Pts Limit: {RECORDED_POSITIONS_LIMIT}",
        f"Predictions: {NUMBER_OF_PREDICTIONS}",
        f"Draw Past: {DRAW_PAST_PREDICTIONS}"
    ]
    pygame.display.set_caption(" | ".join(filter(None, caption_parts)))

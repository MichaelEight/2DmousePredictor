import pygame
import os

# Initialize Pygame
pygame.init()

# Settings
WINDOW_SIZE = (1000, 1000)
DOT_COLOR = (255, 255, 255)
POINT_COLOR = (0, 255, 0)
LINE_COLOR = (0, 255, 255)
TEXT_COLOR = (255, 255, 255)
TEXT_BACKGROUND_COLOR = (0, 0, 0, 128)  # 50% opacity black
TEXT_PADDING = 10
FPS_LIMIT = 30

# Load data files
data_folder_path = 'mouse_data_to_train'
data_files = os.listdir(data_folder_path)
data_files = sorted(data_files)

# Load data from all files
all_data_points = []
for file_name in data_files:
    file_path = os.path.join(data_folder_path, file_name)
    data_points = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(int, line.strip().split(','))
            data_points.append((x, y))
    all_data_points.append((file_name, data_points))

# Initialize variables
current_index = 0

def draw_text_with_background(surface, text, font, color, background_color, pos):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(topleft=pos)
    background_surface = pygame.Surface((text_rect.width, text_rect.height))
    background_surface.set_alpha(background_color[3])  # Set alpha value for transparency
    background_surface.fill(background_color[:3])  # Set RGB values
    surface.blit(background_surface, text_rect.topleft)
    surface.blit(text_surface, text_rect.topleft)

def draw_data_points(WINDOW, data_points, file_name):
    WINDOW.fill((0, 0, 0))
    pygame.draw.circle(WINDOW, DOT_COLOR, (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2), 5)
    for i in range(1, len(data_points)):
        pygame.draw.line(WINDOW, LINE_COLOR, data_points[i - 1], data_points[i], 2)
    for point in data_points:
        pygame.draw.circle(WINDOW, POINT_COLOR, point, 3)

    # Draw text with the file name and number of points
    font = pygame.font.Font(None, 36)
    draw_text_with_background(WINDOW, file_name, font, TEXT_COLOR, TEXT_BACKGROUND_COLOR, (TEXT_PADDING, TEXT_PADDING))
    
    num_points_text = f"Number of points: {len(data_points)}"
    draw_text_with_background(WINDOW, num_points_text, font, TEXT_COLOR, TEXT_BACKGROUND_COLOR, (TEXT_PADDING, TEXT_PADDING + font.get_height() + TEXT_PADDING))

    pygame.display.update()

def main():
    global current_index
    WINDOW = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Data Viewer")

    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    current_index = (current_index + 1) % len(all_data_points)
                elif event.key == pygame.K_LEFT:
                    current_index = (current_index - 1) % len(all_data_points)

        file_name, data_points = all_data_points[current_index]
        draw_data_points(WINDOW, data_points, file_name)
        clock.tick(FPS_LIMIT)

if __name__ == "__main__":
    main()

import pygame
import os
import sys

# Initialize Pygame
pygame.init()

# Settings
WINDOW_SIZE = (1000, 1000)
DOT_COLOR = (255, 255, 255)
POINT_COLOR = (0, 255, 0)
LINE_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)
CENTER_DOT_RADIUS = 5
POINT_RADIUS = 3
TEXT_PADDING = 20

WINDOW = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Mouse Data Viewer")

def draw_data(points):
    WINDOW.fill((0, 0, 0))
    pygame.draw.circle(WINDOW, DOT_COLOR, (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2), CENTER_DOT_RADIUS)

    for i in range(1, len(points)):
        pygame.draw.line(WINDOW, LINE_COLOR, points[i-1], points[i], 2)
    for pos in points:
        pygame.draw.circle(WINDOW, POINT_COLOR, pos, POINT_RADIUS)

    pygame.display.update()

def list_files_and_select():
    files = os.listdir('mouse_data_to_train')
    files = [f for f in files if os.path.isfile(os.path.join('mouse_data_to_train', f))]

    print("[0] Exit")
    for i, file_name in enumerate(files):
        print(f"[{i+1}] {file_name}")

    try:
        choice = int(input("Select a file number: "))
        if choice == 0:
            sys.exit()
        elif 1 <= choice <= len(files):
            return files[choice - 1]
        else:
            print("Invalid choice. Please try again.")
            return list_files_and_select()
    except ValueError:
        print("Invalid input. Please enter a number.")
        return list_files_and_select()

def load_data(file_name):
    points = []
    with open(os.path.join('mouse_data_to_train', file_name), 'r') as file:
        for line in file:
            x, y = map(int, line.strip().split(','))
            points.append((x, y))
    return points

def main():
    file_name = list_files_and_select()
    points = load_data(file_name)
    print(f"Loaded data from {file_name}")

    running = True
    draw_data(points)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

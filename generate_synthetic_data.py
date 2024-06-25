import numpy as np
import random
import math
import os

# Settings
WINDOW_SIZE = (1000, 1000)
DATA_SIZE = 10000  # Number of points to generate
OUTPUT_FILE = 'synthetic_mouse_data.txt'

def generate_sine_wave_data(num_points, amplitude, frequency, phase_shift):
    x = np.linspace(0, 2 * np.pi, num_points)
    y = amplitude * np.sin(frequency * x + phase_shift)
    return list(zip((WINDOW_SIZE[0] / 2 + x * (WINDOW_SIZE[0] / (2 * np.pi))).astype(int),
                    (WINDOW_SIZE[1] / 2 + y * (WINDOW_SIZE[1] / (2 * amplitude))).astype(int)))

def generate_random_walk_data(start_point, num_points, step_size):
    x, y = start_point
    points = [(x, y)]
    for _ in range(num_points - 1):
        angle = random.uniform(0, 2 * np.pi)
        x += step_size * math.cos(angle)
        y += step_size * math.sin(angle)
        x = max(0, min(WINDOW_SIZE[0], x))
        y = max(0, min(WINDOW_SIZE[1], y))
        points.append((int(x), int(y)))
    return points

def generate_random_curve_data(num_points):
    points = []
    x = random.randint(0, WINDOW_SIZE[0])
    y = random.randint(0, WINDOW_SIZE[1])
    for _ in range(num_points):
        x += random.uniform(-5, 5)
        y += random.uniform(-5, 5)
        x = max(0, min(WINDOW_SIZE[0], x))
        y = max(0, min(WINDOW_SIZE[1], y))
        points.append((int(x), int(y)))
    return points

def save_data(points, output_file):
    with open(output_file, 'w') as f:
        for x, y in points:
            f.write(f"{x}, {y}\n")

if __name__ == "__main__":
    all_points = []
    
    # Generate sine wave data
    all_points += generate_sine_wave_data(DATA_SIZE // 3, amplitude=200, frequency=2, phase_shift=0)
    
    # Generate random walk data
    all_points += generate_random_walk_data((WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2), DATA_SIZE // 3, step_size=5)
    
    # Generate random curve data
    all_points += generate_random_curve_data(DATA_SIZE // 3)
    
    # Save generated data
    save_data(all_points, OUTPUT_FILE)
    print(f"Generated {len(all_points)} data points and saved to {OUTPUT_FILE}")

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model_predictor import MousePredictor, save_model
import argparse
import time
import validate_folders_scheme as vfs
from validate_folders_scheme import folders as vfs_folders

vfs.ensure_folders_exist(vfs_folders)

# Define the path to the data folder
data_folder_path = 'data/data_mouse'
trained_model_path = 'models/trained_models'

# Load the recorded data from all files in the folder
def load_data(folder_path, normalize=False, window_size=(1000, 1000), type_of_input='positional'):
    data = []
    files_used = []

    # Debug: Check if the folder exists and list its contents
    if not os.path.exists(folder_path):
        print(f"Folder path does not exist: {folder_path}")
        return data, files_used

    print(f"Reading files from folder: {folder_path}")
    print(f"Contents of the folder: {os.listdir(folder_path)}")

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        files_used.append(file_name)
        print(f"Reading file: {file_path}")  # Debug: Print the file being read
        with open(file_path, 'r') as file:
            points = []
            for line in file:
                try:
                    x, y = map(int, line.strip().split(','))
                    if normalize:
                        x /= window_size[0]
                        y /= window_size[1]
                    points.append((x, y))
                except ValueError:
                    # Skip lines that do not contain valid coordinate pairs
                    continue
            if type_of_input == 'vector' and len(points) > 1:
                vectors = [(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]) for i in range(1, len(points))]
                data.extend(vectors)
            else:
                data.extend(points)
    print(f"Loaded data from files: {files_used}")
    print(f"First 5 data points: {data[:5]}")
    print(f"Total data points loaded: {len(data)}")
    return data, files_used

# Prepare the dataset for training
def prepare_dataset(data, sequence_length, output_size):
    inputs = []
    targets = []
    for i in range(len(data) - sequence_length - output_size + 1):
        input_seq = data[i:i + sequence_length]
        target_seq = data[i + sequence_length:i + sequence_length + output_size]
        if len(input_seq) == sequence_length and len(target_seq) == output_size:
            inputs.append(np.array(input_seq).flatten())
            targets.append(np.array(target_seq).flatten())
    print(f"Prepared {len(inputs)} input sequences")
    return np.array(inputs), np.array(targets)

# Train the model
def train_offline(data, model, criterion, optimizer, sequence_length, output_size, epochs=100):
    inputs, targets = prepare_dataset(data, sequence_length, output_size)
    inputs = torch.FloatTensor(inputs)
    targets = torch.FloatTensor(targets).view(-1, output_size, 2)

    print(f"Training input shape: {inputs.shape}")
    print(f"Training target shape: {targets.shape}")

    if inputs.shape[0] == 0 or targets.shape[0] == 0:
        print("No valid data sequences found. Please check your data preparation.")
        return 0, 0, 0

    start_time = time.time()
    final_loss = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {final_loss}")

    end_time = time.time()
    training_time = end_time - start_time

    return len(inputs), training_time, final_loss

# Create a description file for the model
def create_description_file(sequence_length, output_size, hidden_layers, description, data_size, training_time, final_loss, model, model_path, files_used):
    description_path = model_path.replace('.pth', '_description.txt')
    with open(description_path, 'w') as f:
        f.write(f"Model Name: {os.path.basename(model_path)}\n")
        f.write(f"Sequence Length: {sequence_length}\n")
        f.write(f"Output Size: {output_size}\n")
        f.write(f"Hidden Layers: {hidden_layers}\n")
        f.write(f"Data Size: {data_size}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Final Loss: {final_loss:.4f}\n")
        f.write(f"Description: {description}\n")
        f.write(f"Data Files Used:\n")
        for file in files_used:
            f.write(f"  - {file}\n")
        f.write(f"Model Structure:\n")
        f.write(str(model) + "\n")

# Convert hidden layers to string for naming
def hidden_layers_to_str(hidden_layers):
    return '-'.join([f"{size}{activation[0].upper()}" for size, activation in hidden_layers])

# Main function to train and save the model
def main(sequence_length, output_size, hidden_layers, description, normalize=False, type_of_input='positional'):
    data, files_used = load_data(data_folder_path, normalize=normalize, type_of_input=type_of_input)
    model = MousePredictor(sequence_length, output_size, hidden_layers, type_of_input)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    data_size, training_time, final_loss = train_offline(data, model, criterion, optimizer, sequence_length, output_size)
    norm_flag = "N" if normalize else "U"
    input_type_flag = "VEC" if type_of_input == 'vector' else "POS"
    hidden_layers_str = hidden_layers_to_str(hidden_layers)
    model_path = f"{trained_model_path}/L{sequence_length}_{output_size}_{hidden_layers_str}_{description}_{norm_flag}_{input_type_flag}.pth"
    save_model(model, hidden_layers, model_path)
    create_description_file(sequence_length, output_size, hidden_layers, description, data_size, training_time, final_loss, model, model_path, files_used)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_length', type=int, default=20, help='Length of input sequence')
    parser.add_argument('--output_size', type=int, default=1, help='Number of points to predict')
    parser.add_argument('--hidden_layers', type=str, default="64ReLU-32ReLU", help='Hidden layers configuration')
    parser.add_argument('--desc', type=str, default="mix", help='Describe data used to train model')
    parser.add_argument('--normalize', action='store_true', help='Normalize data coordinates to 0.0-1.0 range')
    parser.add_argument('--type_of_input', type=str, default='positional', choices=['positional', 'vector'], help='Type of input data: positional or vector')
    args = parser.parse_args()

    # Parse hidden layers
    hidden_layers = []
    for hl in args.hidden_layers.split('-'):
        size, activation = int(hl[:-4]), hl[-4:]
        hidden_layers.append((size, activation))

    main(args.sequence_length, args.output_size, hidden_layers, args.desc, args.normalize, args.type_of_input)

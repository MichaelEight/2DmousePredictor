import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from predictor_model import MousePredictor, save_model
import argparse
import time

# Define the path to the data folder
data_folder_path = 'data_mouse'
trained_model_path = 'trained_models'

# Load the recorded data from all files in the folder
def load_data(folder_path, normalize=False, window_size=(1000, 1000)):
    data = []
    files_used = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        files_used.append(file_name)
        with open(file_path, 'r') as file:
            for line in file:
                x, y = map(int, line.strip().split(','))
                if normalize:
                    x /= window_size[0]
                    y /= window_size[1]
                data.append((x, y))
    return data, files_used

# Prepare the dataset for training
def prepare_dataset(data, sequence_length, output_size):
    inputs = []
    targets = []
    for i in range(len(data) - sequence_length - output_size):
        input_seq = data[i:i + sequence_length]
        target_seq = data[i + sequence_length:i + sequence_length + output_size]
        inputs.append(np.array(input_seq).flatten())
        targets.append(np.array(target_seq).flatten())
    return np.array(inputs), np.array(targets)

# Train the model
def train_offline(data, model, criterion, optimizer, sequence_length, output_size, epochs=100):
    inputs, targets = prepare_dataset(data, sequence_length, output_size)
    inputs = torch.FloatTensor(inputs)
    targets = torch.FloatTensor(targets).view(-1, output_size, 2)

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
def main(sequence_length, output_size, hidden_layers, description, normalize=False):
    data, files_used = load_data(data_folder_path, normalize=normalize)
    model = MousePredictor(sequence_length, output_size, hidden_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    data_size, training_time, final_loss = train_offline(data, model, criterion, optimizer, sequence_length, output_size)
    norm_flag = "N" if normalize else "U"
    hidden_layers_str = hidden_layers_to_str(hidden_layers)
    model_path = f"{trained_model_path}/L{sequence_length}_{output_size}_{hidden_layers_str}_{description}_{norm_flag}.pth"
    save_model(model, hidden_layers, model_path)
    create_description_file(sequence_length, output_size, hidden_layers, description, data_size, training_time, final_loss, model, model_path, files_used)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_length', type=int, default=20, help='Length of input sequence')
    parser.add_argument('--output_size', type=int, default=1, help='Number of points to predict')
    parser.add_argument('--hidden_layers', type=str, default="64ReLU-32ReLU", help='Hidden layers configuration')
    parser.add_argument('--desc', type=str, default="mix", help='Describe data used to train model')
    parser.add_argument('--normalize', action='store_true', help='Normalize data coordinates to 0.0-1.0 range')
    args = parser.parse_args()

    # Parse hidden layers
    hidden_layers = []
    for hl in args.hidden_layers.split('-'):
        size, activation = int(hl[:-4]), hl[-4:]
        hidden_layers.append((size, activation))

    main(args.sequence_length, args.output_size, hidden_layers, args.desc, args.normalize)

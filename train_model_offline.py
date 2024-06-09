import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ml_model import MousePredictor, save_model
import argparse
import time

# Define the path to the data folder
data_folder_path = 'mouse_data_to_train'
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
def prepare_dataset(data, sequence_length):
    inputs = []
    targets = []
    for i in range(len(data) - sequence_length):
        input_seq = data[i:i + sequence_length]
        target = data[i + sequence_length]
        inputs.append(np.array(input_seq).flatten())
        targets.append(target)
    return np.array(inputs), np.array(targets)

# Train the model
def train_offline(data, model, criterion, optimizer, sequence_length, epochs=100):
    inputs, targets = prepare_dataset(data, sequence_length)
    inputs = torch.FloatTensor(inputs)
    targets = torch.FloatTensor(targets)

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
def create_description_file(sequence_length, description, data_size, training_time, final_loss, model, model_path, files_used):
    description_path = model_path.replace('.pth', '_description.txt')
    with open(description_path, 'w') as f:
        f.write(f"Model Name: {os.path.basename(model_path)}\n")
        f.write(f"Sequence Length: {sequence_length}\n")
        f.write(f"Data Size: {data_size}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Final Loss: {final_loss:.4f}\n")
        f.write(f"Description: {description}\n")
        f.write(f"Data Files Used:\n")
        for file in files_used:
            f.write(f"  - {file}\n")
        f.write(f"Model Structure:\n")
        f.write(str(model) + "\n")

# Main function to train and save the model
def main(sequence_length, description, normalize=False):
    data, files_used = load_data(data_folder_path, normalize=normalize)
    model = MousePredictor(sequence_length)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    data_size, training_time, final_loss = train_offline(data, model, criterion, optimizer, sequence_length)
    norm_flag = "N" if normalize else "U"
    model_path = f"{trained_model_path}/L{sequence_length}_{description}_{norm_flag}.pth"
    save_model(model, model_path)
    create_description_file(sequence_length, description, data_size, training_time, final_loss, model, model_path, files_used)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_length', type=int, default=20, help='Length of input sequence')
    parser.add_argument('--desc', type=str, default="mix", help='Describe data used to train model')
    parser.add_argument('--normalize', action='store_true', help='Normalize data coordinates to 0.0-1.0 range')
    args = parser.parse_args()
    main(args.sequence_length, args.desc, args.normalize)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ml_model import MousePredictor, save_model
import argparse

# Define the path to the data folder
data_folder_path = 'mouse_data_to_train'

# Load the recorded data from all files in the folder
def load_data(folder_path):
    data = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            for line in file:
                x, y = map(int, line.strip().split(','))
                data.append((x, y))
    return data

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

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Main function to train and save the model
def main(sequence_length):
    data = load_data(data_folder_path)
    model = MousePredictor(sequence_length)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_offline(data, model, criterion, optimizer, sequence_length)
    save_model(model, f"model_{sequence_length}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_length', type=int, default=20, help='Length of input sequence')
    args = parser.parse_args()
    main(args.sequence_length)

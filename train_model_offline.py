import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ml_model import MousePredictor, save_model

# Define the path to the data file
data_file_path = 'data/mouse_positions.txt'

# Load the recorded data
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(int, line.strip().split(','))
            data.append((x, y))
    return data

# Prepare the dataset for training
def prepare_dataset(data, sequence_length=20):
    inputs = []
    targets = []
    for i in range(len(data) - sequence_length):
        input_seq = data[i:i + sequence_length]
        target = data[i + sequence_length]
        inputs.append(np.array(input_seq).flatten())
        targets.append(target)
    return np.array(inputs), np.array(targets)

# Train the model
def train_offline(data, model, criterion, optimizer, epochs=100):
    inputs, targets = prepare_dataset(data)
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
def main():
    data = load_data(data_file_path)
    model = MousePredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_offline(data, model, criterion, optimizer)
    save_model(model)

if __name__ == "__main__":
    main()

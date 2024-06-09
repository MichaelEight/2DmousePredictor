import torch
import torch.nn as nn
import os

# Define the neural network model with a variable input and output size
class MousePredictor(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(MousePredictor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size * 2)  # Predict multiple points

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.output_size, 2)  # Reshape to (batch_size, output_size, 2)

# Training function
def train_model(model, data, target, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

# Save the model
def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)

# Load the model
def load_model(model, path="model.pth"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    return model

# Predict the next points
def predict(model, data):
    model.eval()
    with torch.no_grad():
        return model(data).numpy()

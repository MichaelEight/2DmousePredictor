import torch
import torch.nn as nn
import torch.optim as optim
import os

# Define the neural network model
class MousePredictor(nn.Module):
    def __init__(self):
        super(MousePredictor, self).__init__()
        self.fc1 = nn.Linear(40, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

# Predict the next point
def predict(model, data):
    model.eval()
    with torch.no_grad():
        return model(data).numpy()

import torch
import torch.nn as nn
import os
import numpy as np

# Define the neural network model with a variable input, output size, and hidden layers
class MousePredictor(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_layers=None, type_of_input='positional'):
        super(MousePredictor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers or [(64, 'ReLU'), (32, 'ReLU')]  # Default hidden layers if none are provided
        self.type_of_input = type_of_input

        layers = []
        in_features = input_size * 2

        # Add hidden layers
        for hidden_size, activation in self.hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(getattr(nn, activation)())
            in_features = hidden_size

        # Add output layer
        layers.append(nn.Linear(in_features, output_size * 2))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        print(f"Forward input shape: {x.shape}")
        x = self.model(x)
        print(f"Output shape: {x.shape}")
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
def save_model(model, hidden_layers, path="model.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'hidden_layers': hidden_layers,
        'type_of_input': model.type_of_input
    }
    torch.save(checkpoint, path)

# Load the model
def load_model(input_size, output_size, path="model.pth"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    checkpoint = torch.load(path)
    hidden_layers = checkpoint['hidden_layers']
    type_of_input = checkpoint.get('type_of_input', 'positional')
    model = MousePredictor(input_size, output_size, hidden_layers, type_of_input)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, hidden_layers, type_of_input

# Predict the next points
def predict(model, data):
    model.eval()
    with torch.no_grad():
        return model(data).numpy()

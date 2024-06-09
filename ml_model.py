import torch
import torch.nn as nn
import os

# Define the neural network model with a variable input, output size, and hidden layers
class MousePredictor(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_layers=None):
        super(MousePredictor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers or [(64, 'ReLU'), (32, 'ReLU')]  # Default hidden layers if none are provided

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
        x = self.model(x)
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
        'hidden_layers': hidden_layers
    }
    torch.save(checkpoint, path)

# Load the model
def load_model(input_size, output_size, path="model.pth"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    checkpoint = torch.load(path)
    hidden_layers = checkpoint['hidden_layers']
    model = MousePredictor(input_size, output_size, hidden_layers)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, hidden_layers

# Predict the next points
def predict(model, data):
    model.eval()
    with torch.no_grad():
        return model(data).numpy()

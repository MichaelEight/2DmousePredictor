import torch
import torch.nn as nn
import os

# Define the shape classifier neural network model
class ShapeClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layers=None):
        super(ShapeClassifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers or [(64, 'ReLU'), (32, 'ReLU')]  # Default hidden layers if none are provided

        layers = []
        in_features = input_size * 2

        # Add hidden layers
        for hidden_size, activation in self.hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(getattr(nn, activation)())
            in_features = hidden_size

        # Add output layer
        layers.append(nn.Linear(in_features, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Save the classifier model
def save_model(model, hidden_layers, path="model.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'hidden_layers': hidden_layers
    }
    torch.save(checkpoint, path)

# Load the classifier model
def load_classifier(input_size, num_classes, path="model.pth"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    checkpoint = torch.load(path)
    hidden_layers = checkpoint['hidden_layers']
    class_map = checkpoint['class_map']
    model = ShapeClassifier(input_size, num_classes, hidden_layers)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, hidden_layers, class_map


# Predict the shape
def predict_shape(model, data, sequence_length):
    model.eval()
    with torch.no_grad():
        output = model(data)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
        return predicted.item(), probabilities.squeeze().tolist()


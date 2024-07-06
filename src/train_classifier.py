import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model_shape_classifier import ShapeClassifier, save_model
import argparse
import time
import validate_folders_scheme as vfs
from validate_folders_scheme import folders as vfs_folders

vfs.ensure_folders_exist(vfs_folders)

# Define the path to the data folder
data_folder_path = 'data/data_classifier'
trained_model_path = 'models/trained_models'

# Save the classifier model
def save_model(model, hidden_layers, path="model.pth", class_map=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'hidden_layers': hidden_layers,
        'class_map': class_map,
        'type_of_input': model.type_of_input
    }
    torch.save(checkpoint, path)

# Load the recorded data from all files in the folder
def load_data(folder_path, normalize=False, window_size=(1000, 1000), type_of_input='positional'):
    data = []
    labels = []
    files_used = []
    class_map = {}
    class_counter = 0

    for file_name in os.listdir(folder_path):
        class_name = file_name.split('_')[0]
        if class_name not in class_map:
            class_map[class_name] = class_counter
            class_counter += 1
        
        file_path = os.path.join(folder_path, file_name)
        files_used.append(file_name)
        with open(file_path, 'r') as file:
            sequence = []
            for line in file:
                x, y = map(int, line.strip().split(','))
                sequence.append((x, y))
            if type_of_input == 'vector':
                vectors = [(sequence[i][0] - sequence[i - 1][0], sequence[i][1] - sequence[i - 1][1]) for i in range(1, len(sequence))]
                data.append(vectors)
            else:
                data.append(sequence)
            labels.append(class_map[class_name])
    num_classes = len(class_map)
    return data, labels, files_used, class_map, num_classes

# Prepare the dataset for training
def prepare_dataset(data, labels, sequence_length):
    inputs = []
    targets = []
    for sequence, label in zip(data, labels):
        for i in range(len(sequence) - sequence_length):
            input_seq = sequence[i:i + sequence_length]
            inputs.append(np.array(input_seq).flatten())
            targets.append(label)
    return np.array(inputs), np.array(targets)

# Train the model
def train_classifier(data, labels, model, criterion, optimizer, sequence_length, epochs=100):
    inputs, targets = prepare_dataset(data, labels, sequence_length)
    inputs = torch.FloatTensor(inputs)
    targets = torch.LongTensor(targets)

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
def create_description_file(sequence_length, num_classes, hidden_layers, description, data_size, training_time, final_loss, model, model_path, files_used, class_map):
    description_path = model_path.replace('.pth', '_description.txt')
    with open(description_path, 'w') as f:
        f.write(f"Model Name: {os.path.basename(model_path)}\n")
        f.write(f"Sequence Length: {sequence_length}\n")
        f.write(f"Number of Classes: {num_classes}\n")
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
        f.write(f"Class Map:\n")
        for k, v in class_map.items():
            f.write(f"  - {k}: {v}\n")

# Convert hidden layers to string for naming
def hidden_layers_to_str(hidden_layers):
    return '-'.join([f"{size}{activation[0].upper()}" for size, activation in hidden_layers])

# Main function to train and save the model
def main(sequence_length, hidden_layers, description, normalize=False, type_of_input='positional'):
    data, labels, files_used, class_map, num_classes = load_data(data_folder_path, normalize=normalize, type_of_input=type_of_input)
    model = ShapeClassifier(sequence_length, num_classes, hidden_layers, type_of_input)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    data_size, training_time, final_loss = train_classifier(data, labels, model, criterion, optimizer, sequence_length, epochs=100)
    norm_flag = "N" if normalize else "U"
    input_type_flag = "VEC" if type_of_input == 'vector' else "POS"
    hidden_layers_str = hidden_layers_to_str(hidden_layers)
    model_path = f"{trained_model_path}/classifier_{sequence_length}_{num_classes}_{hidden_layers_str}_{description}_{norm_flag}_{input_type_flag}.pth"
    save_model(model, hidden_layers, model_path, class_map=class_map)
    create_description_file(sequence_length, num_classes, hidden_layers, description, data_size, training_time, final_loss, model, model_path, files_used, class_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_length', type=int, default=20, help='Length of input sequence')
    parser.add_argument('--hidden_layers', type=str, default="64ReLU-32ReLU", help='Hidden layers configuration')
    parser.add_argument('--desc', type=str, default="shapes", help='Describe data used to train model')
    parser.add_argument('--normalize', action='store_true', help='Normalize data coordinates to 0.0-1.0 range')
    parser.add_argument('--type_of_input', type=str, default='positional', choices=['positional', 'vector'], help='Type of input data: positional or vector')
    args = parser.parse_args()

    # Parse hidden layers
    hidden_layers = []
    for hl in args.hidden_layers.split('-'):
        size, activation = int(hl[:-4]), hl[-4:]
        hidden_layers.append((size, activation))

    main(args.sequence_length, hidden_layers, args.desc, args.normalize, args.type_of_input)


# Mouse Predictor and Shape Classifier

This project allows you to draw shapes with your mouse and then train AI/ML models for shape classification and/or mouse path prediction based on previous positions.

## Table of Contents

- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Launching](#launching)
- [Start Simulation](#start-simulation)
  - [Settings](#settings)
    - [Predictor Definition](#predictor-definition)
    - [Classifier Definition](#classifier-definition)
  - [Simulation](#simulation)
    - [Shortcuts](#shortcuts)
    - [Recommended for Training](#recommended-for-training)
  - [Finishing](#finishing)
- [Training](#training)
  - [Settings](#settings)
    - [Data Files](#data-files)
    - [Model Parameters](#model-parameters)
  - [Training Process](#training-process)
- [Data Viewer](#data-viewer)
- [Manual Operations](#manual-operations)
  - [Training Predictor](#training-predictor)
  - [Training Classifier](#training-classifier)

## Getting Started

### Requirements

Install dependencies:

```sh
pip install numpy torch customtkinter pygame
```

### Launching

Run `launch.py` for GUI. You should see the menu:

<img src="https://github.com/MichaelEight/2DmousePredictor/assets/56772277/73d0dc5c-85e7-4453-8656-24d89b6ff51f" width="50%" height="50%" />

## Start Simulation

### Settings

Click **Start Simulation**. You should see:

<img src="https://github.com/MichaelEight/2DmousePredictor/assets/56772277/7328f88c-736f-4b20-93f7-8e381eff5714" width="50%" height="50%" />

You can select multiple Predictors and a single Classifier or none.

#### Predictor Definition

Predictor models predict the future path of the mouse based on previous positions.

**Naming Convention:** `Laa_bb_ccR-ddR-eeR_desc_f.pth`

- `L`: Indicator it's a Predictor model
- `aa`: Model input size (number of points `(x, y)` pairs)
- `bb`: Model output size (recommended to keep at 1 for recursion)
- `cc`, `dd`, `ee`: Sizes of hidden layers (set to 0 to skip)
- `R`: Activation function for layer (modifiable via command line)
- `desc`: Description of what the model is trained on
- `f`: Flag indicating if data is normalized (`N` for normalized, `U` for not)

#### Classifier Definition

Classifier models classify shapes and display the probability of each shape being currently drawn.

**Naming Convention:** `classifier_aa_bb_ccR-ddR-eeR-desc_f.pth`

- `classifier`: Indicator it's a Classifier model
- `aa`: Model input size (number of points `(x, y)` pairs)
- `bb`: Model output size (number of recognized shapes)
- `cc`, `dd`, `ee`: Sizes of hidden layers (set to 0 to skip)
- `R`: Activation function for layer (modifiable via command line)
- `desc`: Description of what the model is trained on
- `f`: Flag indicating if data is normalized (`N` for normalized, `U` for not)

### Simulation

After pressing **Start**, you will see:

<img src="https://github.com/MichaelEight/2DmousePredictor/assets/56772277/044fc75c-fed3-471a-bad7-a07cf6e117a9" width="50%" height="50%" />

You can move your mouse to draw and adjust simulation settings via shortcuts.

#### Shortcuts

- `O`: Toggle current prediction
- `P`: Toggle ghost of last prediction
- `N/M`: Decrease/increase length of mouse path
- `K/L`: Decrease/increase length of prediction
- `,/.`: Decrease/increase FPS (refresh rate)
- `C`: Toggle continuous update (update every frame if `TRUE`, only when mouse moves if `FALSE`)

#### Recommended for Training

- Choose a shape (circle, square, etc.) and stick to it.
- Choose a direction (clockwise/anticlockwise) and stick to it.
- The longer you do this, the more data is collected for training.

### Finishing

Press `ESC` to save collected data (mouse positions) to a file for later training.

<img src="https://github.com/MichaelEight/2DmousePredictor/assets/56772277/57102183-96f7-4771-99d7-465ee22d7a1d" width="25%" height="25%" />

## Training

### Settings

Click **Start Simulation**. You should see:

<img src="https://github.com/MichaelEight/2DmousePredictor/assets/56772277/c5b435ba-fec3-4f85-b90f-8e57760353a2" width="50%" height="50%" />

#### Data Files

These are files saved from the simulation.

**Important:** Correct data naming is crucial for the classifier! The name should start with the classified object name followed by an underscore (`_`). For example: `square_somethingsomething.txt`. The classifier's output size is determined by the number of unique object names. For example, if you have `square_1`, `square_2`, `circle_abc_123`, `circle_1`, you will have two unique objects: "square" and "circle".

#### Model Parameters

- **Training Type:** Select if you want to train a Predictor Model or Classifier Model.
- **Hidden Layers:** Select sizes (number of nodes) in each hidden layer.
- **Input Size:** Number of points (so `(x, y)` pairs) taken as input.
- **Output Size:** (Predictor only) Number of points to be predicted by default. **Tip:** For a small amount of data, keep the output = 1.
- **Description:** A short description for the file name to track what data the model was trained on.
- **Normalize:** Toggle data normalization. Instead of inputs being `0 <= x <= 1000` (window size), they will be in the range `0.0 <= x <= 1.0`.

### Training Process

The training process will run in the console (as a Python process). You will be informed when it finishes:

<img src="https://github.com/MichaelEight/2DmousePredictor/assets/56772277/4e8fa94d-4b5b-4228-998a-10be5ca6f839" width="25%" height="25%" />

The model can be used immediately in the simulation.

## Data Viewer

The Data Viewer loads all data files and allows you to view them. It shows the number of points stored in each file. Use the left/right arrows to navigate through the files.

<img src="https://github.com/MichaelEight/2DmousePredictor/assets/56772277/7493a5f7-f79a-47c2-97fd-362fbe98ffaa" width="50%" height="50%" />
<img src="https://github.com/MichaelEight/2DmousePredictor/assets/56772277/355d8256-cb04-4633-bacd-090369edc305" width="50%" height="50%" />

## Manual Operations

### Training Predictor

```sh
python train_predictor.py
```

**Arguments:**

- `--sequence_length`: Model input, number of points used for prediction (default: 20).
- `--output_size`: Number of points to predict (default: 1). **Note:** If the app displays 1 prediction and the model is trained for 5, it will not show up until you increase the number of displayed predictions.
- `--hidden_layers`: Configuration of hidden layers (default: "64ReLU-32ReLU").
- `--desc`: Short description for the model (default: "mix").
- `--normalize`: Normalize data coordinates to the 0.0-1.0 range (optional).

**Example:**

```sh
python train_predictor.py --sequence_length 20 --output_size 1 --hidden_layers "64ReLU-32ReLU" --desc mix --normalize
```

**Default (Recommended) Command:**

```sh
python train_predictor.py --sequence_length 20 --output_size 1 --hidden_layers "64ReLU-32ReLU" --desc mix
```

### Training Classifier

```sh
python train_classifier.py
```

**Arguments:**

- `--sequence_length`: Model input, number of points used for prediction (default: 20).
- `--hidden_layers`: Configuration of hidden layers (default: "64ReLU-32ReLU").
- `--desc`: Short description for the model (default: "CirclesSquares").
- `--normalize`: Normalize data coordinates to the 0.0-1.0 range (optional).

**Example:**

```sh
python train_classifier.py --sequence_length 20 --hidden_layers "64ReLU-32ReLU" --desc CirclesSquares --normalize
```

**Default (Recommended) Command:**

```sh
python train_classifier.py --sequence_length 20 --hidden_layers "64ReLU-32ReLU" --desc CirclesSquares
```

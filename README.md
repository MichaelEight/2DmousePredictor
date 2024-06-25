# Mouse Predictor and Shape Classifier
Just a simple project. It lets you draw shapes with your mouse and then train AI/ML models for shape classification (classifier model) and/or mouse path prediction based on previous positions (predictor model).

## Table of Contents

- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Launching](#launching)
- [Start Simulation](#start-simulation)
  - [Settings](#settings)
    - [Predictor definition](#predictor-definition)
    - [Classifier definition](#classifier-definition)
  - [Simulation](#simulation)
    - [Shortcuts](#shortcuts)
    - [Recommended for training](#recommended-for-training)
  - [Finishing](#finishing)
- [Training](#training)
  - [Settings](#settings)
    - [Data Files](#data-files)
    - [Model Parameters](#model-parameters)
  - [Training Process](#training-process)
- [Data Viewer](#data-viewer)



# Getting Started

## Requirements

Install dependencies: ```python pip install numpy torch customtkinter pygame```

## Launching
Run `launch.py` for GUI. You should be welcomed with a menu:

![image](https://github.com/MichaelEight/2DmousePredictor/assets/56772277/73d0dc5c-85e7-4453-8656-24d89b6ff51f)

## Start Simulation

### Settings
Click `Start Simulation`. You should see:

![image](https://github.com/MichaelEight/2DmousePredictor/assets/56772277/7328f88c-736f-4b20-93f7-8e381eff5714)

You can select multiple Predictors and a single Classifier... or none.

**Predictor** - model used to predict future path of mouse based on previous positions.

**Naming:** Laa_bb_ccR-ddR-eeR_desc_f.pth, where:

- L - indicator it's Predictor model
- aa - model input size, so how many points (i.e. (x,y) pairs) are inputted into the model
- bb - model output size, so how many points are outputted (predicted). RECOMMENDED to keep at 1, because app uses "recursion" to get next points (predicted point is treated as last position, where mouse was and fed back to the model).
- cc, dd, ee - sizes of hidden layers. Can be set at 0 to skip hidden layer.
- R - activator function for layer. Currently it's not available to change it in GUI, only via command.
- desc - short description to keep track, what is this model trained on... or whatever you want
- f - flag used to indicated if data is normalized (N) or not normalized (U)

**Classifier** - model uesd to classify shapes and display probability of each of them being currently drawn

**Naming:** classifier_aa_bb_ccR-ddR-eeR-desc_f.pth

- classifier - indicator it's Classifier model... duh
- aa - model input size, so how many points (i.e. (x,y) pairs) are inputted into the model
- bb - model output size, so how many different shapes are recognized
- cc, dd, ee - sizes of hidden layers. Can be set at 0 to skip hidden layer.
- R - activator function for layer. Currently it's not available to change it in GUI, only via command.
- desc - short description to keep track, what is this model trained on... or whatever you want
- f - flag used to indicated if data is normalized (N) or not normalized (U)

### Simulation

After pressing `Start`, you will be moved to the following window:

![image](https://github.com/MichaelEight/2DmousePredictor/assets/56772277/044fc75c-fed3-471a-bad7-a07cf6e117a9)

There you can move around your mouse and draw. You can adjust simulation settings via shortcuts.

#### Shortcuts

- O   - toggle current prediction
- P   - toggle ghost of last prediction
- N/M - decrease/increase length of mouse path
- K/L - decrease/increase length of prediction
- ,/. - decrease/increase FPS (refresh rate)
- C   - toggle continuous update (FALSE = Update only when mouse moves, TRUE = Update every frame)

#### Recommended for training

- choose a shape (circle, square etc.) and stick to it
- choose a direction (clockwise/anticlockwise) and stick to it
- The longer you do this, the more data is collected for training

### Finishing

When pressed ESC, you will be prompted to save collected data (mouse positions) to file. It can be used for training models later on.

![image](https://github.com/MichaelEight/2DmousePredictor/assets/56772277/57102183-96f7-4771-99d7-465ee22d7a1d)

## Training

### Settings
Click `Start Simulation`. You should see:

![image](https://github.com/MichaelEight/2DmousePredictor/assets/56772277/c5b435ba-fec3-4f85-b90f-8e57760353a2)

#### Data Files
These are files saved from the simulation.

**IMPORTANT! Correct data naming is important for classifier!** It should always starts with the name of classified object, followed by a '_'. Example: `square_somethingsomething.txt`. Classifier's output size is determined by amount of unique object names. So if you select: square_1, square_2, circle_abc_123, circle_1; you will have just 2 unique objects: "square" and "circle". Data naming for predictor doesn't matter, however it's a good practice to make the names meaningful. 

#### Model Parameters
Training Type - Select if you want to train Predictor Model or Classifier Model.

Hidden Layers - Select sizes (amount of nodes) in each hidden layer.

Input Size - How many points (so (x,y) pairs) should be taken as an input.

Output Size - (Predictor only) How many points should be predicted by default (**TIP:** for little amount of data keep the output = 1. Tested on 20k points or less)

Description - Just a short description, which will be inserted to the file name, so you can keep track e.g. on what data was the model trained.

Normalize - Toggle data normalization. Instead of inputs being 0 <= x <= 1000 (window size), they will be packed into range 0.0 <= x <= 1.0.


### Training Process

Training process will be run in the console (it's a Python process after all). You will be informed at the end:

![image](https://github.com/MichaelEight/2DmousePredictor/assets/56772277/4e8fa94d-4b5b-4228-998a-10be5ca6f839)

You should be able to use the model right away in the simulation.


## Data Viewer

As the name suggests, it loads all data files and allows you to view them. It shows the amount of points stored in the file. You can press left/right arrows to move to the next file.

![image](https://github.com/MichaelEight/2DmousePredictor/assets/56772277/7493a5f7-f79a-47c2-97fd-362fbe98ffaa)

![image](https://github.com/MichaelEight/2DmousePredictor/assets/56772277/355d8256-cb04-4633-bacd-090369edc305)


## Manual Operations

### Training Predictor

  `python train_predictor.py`  
  
  Arguments:
  - `--sequence_length`: Model input, how many points are used for prediction (default: 20).
  - `--output_size`: Number of points to predict (default: 1).
  NOTE: if the app is set to display 1 predictions and model is trained for 5, it will not show up until you increase the number of displayed predictions!
  - `--hidden_layers`: Configuration of hidden layers (default: "64ReLU-32ReLU").
  - `--desc`: Short description to see, what is this model (default: "mix").
  - `--normalize`: Normalize data coordinates to the 0.0-1.0 range (optional).
  - Example:
    
    `python train_predictor.py --sequence_length 20 --output_size 1 --hidden_layers "64ReLU-32ReLU" --desc mix --normalize`
    
  - Default (Recommended) Command:
  
    `python train_predictor.py --sequence_length 20 --output_size 1 --hidden_layers "64ReLU-32ReLU" --desc mix`

### Training Classifier

  `python train_classifier.py`
  
  Arguments:
  - `--sequence_length`: Model input, how many points are used for prediction (default: 20).
  - `--hidden_layers`: Configuration of hidden layers (default: "64ReLU-32ReLU").
  - `--desc`: Short description to see, what data was used (default: "CirclesSquares").
  - `--normalize`: Normalize data coordinates to the 0.0-1.0 range (optional).
  - Example:
    
    `python train_classifier.py --sequence_length 20 --hidden_layers "64ReLU-32ReLU" --desc CirclesSquares --normalize`
    
  - Default (Recommended) Command:
  
    `python train_classifier.py --sequence_length 20 --hidden_layers "64ReLU-32ReLU" --desc CirclesSquares`

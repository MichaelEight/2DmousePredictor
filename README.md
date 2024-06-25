# Mouse Predictor and Shape Classifier
Just a simple project. Let's you draw with your mouse and train AI/ML models for shape classification (classifier model) and mouse path prediction based on previous positions (predictor model).

# How to use it?

## Requirements

Install dependencies: ```python pip install numpy torch customtkinter pygame```

## Launching
**(NEW)** Run `launch.py` for GUI. You should be welcomed with a menu:

![image](https://github.com/MichaelEight/2DmousePredictor/assets/56772277/73d0dc5c-85e7-4453-8656-24d89b6ff51f)

## Start Simulation - Settings
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

## Start Simulation - Simulation

After pressing `Start`, you will be moved to the following window:

![image](https://github.com/MichaelEight/2DmousePredictor/assets/56772277/044fc75c-fed3-471a-bad7-a07cf6e117a9)

There you can move around your mouse and draw. You can adjust simulation settings via shortcuts.

### Shortcuts

- O   - toggle current prediction
- P   - toggle ghost of last prediction
- N/M - decrease/increase length of mouse path
- K/L - decrease/increase length of prediction
- ,/. - decrease/increase FPS (refresh rate)
- C   - toggle continuous update (FALSE = Update only when mouse moves, TRUE = Update every frame)

### Recommended for training

- choose a shape (circle, square etc.) and stick to it
- choose a direction (clockwise/anticlockwise) and stick to it
- The longer you do this, the more data is collected for training

## Start Simulation - Finishing

When pressed ESC, you will be prompted to save collected data (mouse positions) to file. It can be used for training models later on.

![image](https://github.com/MichaelEight/2DmousePredictor/assets/56772277/57102183-96f7-4771-99d7-465ee22d7a1d)





(if you don't have any mouse data yet)
1. Run main.py
2. Move mouse around the screen
Recommended: choose a shape (circle, square etc.) and stick to it.
Recommended: choose a direction (clockwise/anticlockwise) and stick to it.
Recommended: The longer you do this, the more data is collected for training
3. Press ESC to quit
4. Open folder data/
5. Rename mouse_positions.txt
Recommended: {shape}_{description}.txt
Example:     square_SmallClk.txt
6. Move the file to:
a) data_mouse/ to train prediction model
b) data_classifier/ to train classifier model

(if you have mouse data)
7. Run model training - see instruction for train_predictor.py and train_classifier.py

(if you have trained model)
INFO: Only *.pth files are models! .txt files are just model descriptions!
8. Move any number of predictor models to models_to_load/
NOTE: trained models are saved in trained_models/
9. Move 1 classifier model to models_to_load/

NOTE: App will still work without predictors and classifier models 

INFO: TITLE OF THE APP HOLDS INFO ABOUT SIMULATION
FPS, Continuous updates, length of mouse path, length of prediction, is ghost of past prediction toggled and how many mouse points are recorded so far. 


# Instructions for Python Scripts

1. main.py
- Purpose: Main app. Visualize and predict mouse movements, recognize shapes the mouse is making.
- How to Run:
  python main.py

Shortcuts:
O   - toggle current prediction
P   - toggle ghost of last prediction
N/M - decrease/increase length of mouse path
K/L - decrease/increase length of prediction
,/. - decrease/increase FPS (refresh rate)
C   - toggle continuous update (FALSE = Update only when mouse moves, TRUE = Update every frame)
  
3. train_predictor.py
- Purpose: This script trains a mouse movement prediction model using recorded mouse data.
- How to Run:
  
  python train_predictor.py
  
  Arguments:
  - `--sequence_length`: Model input, how many points are used for prediction (default: 20).
  - `--output_size`: Number of points to predict (default: 1).
  NOTE: if the app is set to display 1 predictions and model is trained for 5, it will not show up until you increase the number of displayed predictions!
  - `--hidden_layers`: Configuration of hidden layers (default: "64ReLU-32ReLU").
  - `--desc`: Short description to see, what is this model (default: "mix").
  - `--normalize`: Normalize data coordinates to the 0.0-1.0 range (optional).
  - Example:
    
    python train_predictor.py --sequence_length 20 --output_size 1 --hidden_layers "64ReLU-32ReLU" --desc mix --normalize
    
  Default (Recommended) Command:
  
  python train_predictor.py --sequence_length 20 --output_size 1 --hidden_layers "64ReLU-32ReLU" --desc mix
  

4. train_classifier.py
- Purpose: This script trains a shape classifier model using recorded shape data.
- How to Run:
  
  python train_classifier.py
  
  Arguments:
  - `--sequence_length`: Model input, how many points are used for prediction (default: 20).
  - `--hidden_layers`: Configuration of hidden layers (default: "64ReLU-32ReLU").
  - `--desc`: Short description to see, what data was used (default: "CirclesSquares").
  - `--normalize`: Normalize data coordinates to the 0.0-1.0 range (optional).
  - Example:
    
    python train_classifier.py --sequence_length 20 --hidden_layers "64ReLU-32ReLU" --desc CirclesSquares --normalize
    
  Default (Recommended) Command:
  
  python train_classifier.py --sequence_length 20 --hidden_layers "64ReLU-32ReLU" --desc CirclesSquares
  

5. mouse_data_viewer.py
- Purpose: This script visualizes the recorded mouse movements 
- How to Run:
  
  python mouse_data_viewer.py
  
 You can use left/right arrows to view previous/next data files
 
 

# Hidden Layers Settings
DEFAULT: 64ReLU-32ReLU
Meaning: 2 hidden layers, first with 64 nodes, second with 32 nodes. ReLU - activation function.
Can be expanded to 3 hidden layers e.g. 64ReLU-32ReLU-16ReLU

Simple explanation:

64ReLU-32ReLU:
+ needs small amount of data to work (under 10k mouse points should be sufficient)
+ relatively good
- ... but only for short predictions

128ReLU-64ReLU:
+ more accurate than 64-32
+ can handle a little bit longer predictions
- requires more data (guesstimate - 50k should be enough)

128ReLU-64ReLU-32ReLU:
+ much more accurate (not proven. Lack of data)
- requires tons of data (100k+)

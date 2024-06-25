# How to use?

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
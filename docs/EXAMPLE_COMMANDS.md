TRAIN MODEL
python train_predictor.py --sequence_length 20 --output_size 1 --desc mix
python train_predictor.py --sequence_length 20 --output_size 1 --hidden_layers 64ReLU-32ReLU --desc mix
python train_predictor.py --sequence_length 20 --output_size 5 --desc mix --normalize

TRAIN CLASSIFIER
python train_classifier.py --sequence_length 20 --hidden_layers 64ReLU-32ReLU --desc shapes
python train_classifier.py --sequence_length 20 --hidden_layers 64ReLU-32ReLU --desc shapes --normalize
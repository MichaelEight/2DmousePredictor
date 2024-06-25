import os

folders = [
    'data',
    'data/data_classifier',
    'data/data_mouse',
    'data/data_queue',
    'models/models_to_load',
    'models/trained_models'
]

def ensure_folders_exist(folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Folder created: {folder}")

# Ensure folders exist once at the start
ensure_folders_exist(folders)

import os

# List of folders to ensure they exist
folders = [
    "data/",
    "data/errors/",
    "data_classifier/",
    "data_mouse/",
    "data_queue/",
    "models_to_load/",
    "trained_models/",
    "trained_models/archive/"
]

# Function to ensure each folder exists
def ensure_folders_exist(folder_list):
    for folder in folder_list:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
        else:
            print(f"Folder already exists: {folder}")

# Call the function with the list of folders
ensure_folders_exist(folders)
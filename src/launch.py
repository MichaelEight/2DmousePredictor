import os
import subprocess
import shutil
import customtkinter as ctk
from tkinter import messagebox, simpledialog, StringVar

import validate_folders_scheme as vfs
from validate_folders_scheme import folders as vfs_folders

vfs.ensure_folders_exist(vfs_folders)

# Constants for input validation
INPUT_SIZE_MIN = 1
INPUT_SIZE_MAX = 999
OUTPUT_SIZE_MIN = 1
OUTPUT_SIZE_MAX = 999

# Current version and author information
CURRENT_VERSION = "2.0.0"
AUTHOR = "Michael Eight"

# Helper functions
def get_pth_files(folder):
    return [f for f in os.listdir(folder) if f.endswith('.pth')]

def get_txt_files(folder):
    return [f for f in os.listdir(folder) if f.endswith('.txt')]

def count_lines(filepath):
    with open(os.path.join('data/data_queue', filepath), 'r') as file:
        return len(file.readlines())

def clear_mouse_positions_file():
    with open('data/data_mouse/mouse_positions.txt', 'w') as file:
        file.truncate(0)

def append_mouse_position(position):
    with open('data/data_mouse/mouse_positions.txt', 'a') as file:
        file.write(f"{position[0]},{position[1]}\n")

def launch_main(models_path, selected_models, selected_classifier):
    args = ["python", "src/main.py", "--models_path", models_path]
    if selected_models:
        args += ["--models"] + selected_models
    if selected_classifier:
        args += ["--classifier", selected_classifier]
    process = subprocess.Popen(args)
    return process

def launch_train_predictor(args):
    process = subprocess.Popen(["python", "src/train_predictor.py"] + args)
    return process

def launch_train_classifier(args):
    process = subprocess.Popen(["python", "src/train_classifier.py"] + args)
    return process

def validate_integer_input(new_value, min_value, max_value):
    if new_value.isdigit():
        value = int(new_value)
        if min_value <= value <= max_value:
            return True
    return False

# Main window
class MainWindow:
    def __init__(self, master):
        self.master = master
        master.title("Main Menu")
        master.geometry("500x300")
        master.eval('tk::PlaceWindow . center')

        ctk.CTkLabel(master, text="Mouse Prediction App", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)

        self.start_simulation_button = ctk.CTkButton(master, text="Start Simulation", command=self.start_simulation)
        self.start_simulation_button.pack(pady=10)

        self.train_button = ctk.CTkButton(master, text="Training", command=self.train)
        self.train_button.pack(pady=10)

        self.data_viewer_button = ctk.CTkButton(master, text="Data Viewer", command=self.data_viewer)
        self.data_viewer_button.pack(pady=10)

        self.about_button = ctk.CTkButton(master, text="About", command=self.open_about)
        self.about_button.pack(pady=10)

    def start_simulation(self):
        self.simulation_window = SimulationWindow(self.master)

    def train(self):
        self.train_window = TrainWindow(self.master)

    def data_viewer(self):
        subprocess.Popen(["python", "src/mouse_data_viewer.py"])

    def open_about(self):
        self.about_window = AboutWindow(self.master)

class AboutWindow:
    def __init__(self, master):
        self.top = ctk.CTkToplevel(master)
        self.top.title("About")
        self.top.geometry("300x200")
        self.center_window(self.top)
        self.top.lift()  # Ensure the new window is on top
        self.top.focus_force()  # Force focus on the new window

        ctk.CTkLabel(self.top, text="About", font=ctk.CTkFont(size=18)).pack(pady=10)
        ctk.CTkLabel(self.top, text=f"Author: {AUTHOR}", font=ctk.CTkFont(size=14)).pack(pady=10)
        ctk.CTkLabel(self.top, text=f"Version: {CURRENT_VERSION}", font=ctk.CTkFont(size=14)).pack(pady=10)

    def center_window(self, window):
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        window.geometry(f'{width}x{height}+{x}+{y}')

class SimulationWindow:
    def __init__(self, master):
        self.top = ctk.CTkToplevel(master)
        self.top.title("Start Simulation")
        self.top.geometry("600x800")
        self.top.minsize(600, 800)
        self.center_window(self.top)
        self.top.lift()  # Ensure the new window is on top
        self.top.attributes("-topmost", True)
        self.top.focus_force()  # Force focus on the new window

        self.top.bind("<Escape>", lambda e: self.close_window())  # Bind ESC to close

        self.all_files = get_pth_files('models/trained_models')
        self.classifier_files = [f for f in self.all_files if f.startswith('classifier')]
        self.predictor_files = [f for f in self.all_files if not f.startswith('classifier')]

        ctk.CTkLabel(self.top, text="Select Predictor Models:", font=ctk.CTkFont(size=18)).pack(pady=10)
        self.predictor_vars = {}
        for file in self.predictor_files:
            var = StringVar(value="0")
            chk = ctk.CTkCheckBox(self.top, text=file, variable=var, onvalue="1", offvalue="0")
            chk.pack(anchor='w', padx=20)
            self.predictor_vars[file] = var

        ctk.CTkLabel(self.top, text="Select Classifier Model:", font=ctk.CTkFont(size=18)).pack(pady=10)
        self.classifier_var = StringVar(value="None")
        ctk.CTkRadioButton(self.top, text="None", variable=self.classifier_var, value="None").pack(anchor='w', padx=20)
        for file in self.classifier_files:
            ctk.CTkRadioButton(self.top, text=file, variable=self.classifier_var, value=file).pack(anchor='w', padx=20)

        self.start_button = ctk.CTkButton(self.top, text="Start", command=self.start_simulation)
        self.start_button.pack(pady=20)

    def center_window(self, window):
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        window.geometry(f'{width}x{height}+{x}+{y}')

    def start_simulation(self):
        clear_mouse_positions_file()  # Clear the file at the start of the simulation

        selected_predictors = [file for file, var in self.predictor_vars.items() if var.get() == "1"]
        selected_classifier = self.classifier_var.get() if self.classifier_var.get() != "None" else None

        process = launch_main('models/models_to_load', selected_predictors, selected_classifier)
        self.top.attributes("-topmost", False)  # Remove the always-on-top attribute

        self.top.withdraw()  # Hide the settings window while the simulation is running
        self.top.after(100, self.check_simulation, process)

    def check_simulation(self, process):
        retcode = process.poll()
        if retcode is not None:
            self.top.deiconify()  # Show the settings window again
            self.ask_save_data()
        else:
            self.top.after(100, self.check_simulation, process)

    def ask_save_data(self):
        if messagebox.askyesno("Save Data", "Do you want to save mouse data?"):
            new_name = simpledialog.askstring("Input", "Enter new file name:", parent=self.top)
            if new_name:
                src_file = 'data/data_mouse/mouse_positions.txt'
                dst_file = f'data/data_queue/{new_name}.txt'
                if os.path.exists(src_file):
                    shutil.move(src_file, dst_file)
                else:
                    messagebox.showerror("File Not Found", f"File '{src_file}' not found. Data could not be saved.")
        self.close_window()

    def close_window(self):
        self.top.destroy()

# Main window
class MainWindow:
    def __init__(self, master):
        self.master = master
        master.title("Main Menu")
        master.geometry("500x300")
        master.eval('tk::PlaceWindow . center')

        ctk.CTkLabel(master, text="Mouse Prediction App", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)

        self.start_simulation_button = ctk.CTkButton(master, text="Start Simulation", command=self.start_simulation)
        self.start_simulation_button.pack(pady=10)

        self.train_button = ctk.CTkButton(master, text="Training", command=self.train)
        self.train_button.pack(pady=10)

        self.data_viewer_button = ctk.CTkButton(master, text="Data Viewer", command=self.data_viewer)
        self.data_viewer_button.pack(pady=10)

        self.about_button = ctk.CTkButton(master, text="About", command=self.open_about)
        self.about_button.pack(pady=10)

    def start_simulation(self):
        self.simulation_window = SimulationWindow(self.master)

    def train(self):
        self.train_window = TrainWindow(self.master)

    def data_viewer(self):
        subprocess.Popen(["python", "src/mouse_data_viewer.py"])

    def open_about(self):
        self.about_window = AboutWindow(self.master)

class TrainWindow:
    def __init__(self, master):
        self.top = ctk.CTkToplevel(master)
        self.top.title("Training")
        self.top.geometry("600x800")
        self.top.minsize(600, 800)
        self.center_window(self.top)
        self.top.lift()  # Ensure the new window is on top
        self.top.attributes("-topmost", True)
        self.top.focus_force()  # Force focus on the new window

        self.top.bind("<Escape>", lambda e: self.close_window())  # Bind ESC to close

        self.data_files = get_txt_files('data/data_queue')  # Corrected path

        ctk.CTkLabel(self.top, text="Select Data Files:", font=ctk.CTkFont(size=18)).pack(pady=10)
        self.data_vars = {}
        for file in self.data_files:
            var = StringVar()
            chk = ctk.CTkCheckBox(self.top, text=f"{file} ({count_lines(file)})", variable=var)
            chk.pack(anchor='w', padx=20)
            self.data_vars[file] = var

        ctk.CTkLabel(self.top, text="Training Type:", font=ctk.CTkFont(size=18)).pack(pady=10)
        self.train_type = StringVar(value="predictor")
        ctk.CTkRadioButton(self.top, text="Predictor", variable=self.train_type, value="predictor").pack(anchor='w', padx=20)
        ctk.CTkRadioButton(self.top, text="Classifier", variable=self.train_type, value="classifier").pack(anchor='w', padx=20)

        ctk.CTkLabel(self.top, text="Hidden Layers:", font=ctk.CTkFont(size=18)).pack(pady=10)
        self.hidden_layers = []
        default_hidden_layers = ["64", "32", "none"]
        for i in range(3):
            var = StringVar(value=default_hidden_layers[i])
            self.hidden_layers.append(var)
            options = ["none", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"]
            dropdown = ctk.CTkComboBox(self.top, values=options, variable=var)
            dropdown.pack(padx=20, pady=5)

        validate_cmd = (self.top.register(lambda new_value: validate_integer_input(new_value, INPUT_SIZE_MIN, INPUT_SIZE_MAX)), "%P")

        ctk.CTkLabel(self.top, text="Input Size:", font=ctk.CTkFont(size=18)).pack(pady=10)
        self.input_size = ctk.CTkEntry(self.top, width=200, validate="key", validatecommand=validate_cmd)
        self.input_size.insert(0, "20")
        self.input_size.pack(pady=5)

        ctk.CTkLabel(self.top, text="Output Size:", font=ctk.CTkFont(size=18)).pack(pady=10)
        self.output_size = ctk.CTkEntry(self.top, width=200, validate="key", validatecommand=validate_cmd)
        self.output_size.insert(0, "1")
        self.output_size.pack(pady=5)

        ctk.CTkLabel(self.top, text="Description:", font=ctk.CTkFont(size=18)).pack(pady=10)
        self.description = ctk.CTkEntry(self.top, width=200)
        self.description.pack(pady=5)

        self.normalize = StringVar()
        ctk.CTkCheckBox(self.top, text="Normalize", variable=self.normalize).pack(pady=10)

        ctk.CTkLabel(self.top, text="Input Type:", font=ctk.CTkFont(size=18)).pack(pady=10)
        self.input_type = StringVar(value="positional")
        ctk.CTkRadioButton(self.top, text="Positional", variable=self.input_type, value="positional").pack(anchor='w', padx=20)
        ctk.CTkRadioButton(self.top, text="Vector", variable=self.input_type, value="vector").pack(anchor='w', padx=20)

        self.train_button = ctk.CTkButton(self.top, text="Train", command=self.start_training)
        self.train_button.pack(pady=20)

    def center_window(self, window):
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        window.geometry(f'{width}x{height}+{x}+{y}')

    def start_training(self):
        selected_data = [file for file, var in self.data_vars.items() if var.get() == "1"]
        hidden_layers_str = '-'.join([f"{size.get()}ReLU" for size in self.hidden_layers if size.get() != "none"])

        self.args = [
            '--sequence_length', self.input_size.get(),
            '--hidden_layers', hidden_layers_str,
            '--desc', self.description.get(),
            '--type_of_input', self.input_type.get()
        ]
        if self.normalize.get():
            self.args.append('--normalize')

        if self.train_type.get() == 'predictor':
            self.args.extend(['--output_size', self.output_size.get()])
            process = launch_train_predictor(self.args)
        else:
            process = launch_train_classifier(self.args)

        self.top.attributes("-topmost", False)  # Remove the always-on-top attribute

        self.top.withdraw()  # Hide the settings window while training is running
        self.top.after(100, self.check_training, process)

    def check_training(self, process):
        retcode = process.poll()
        if retcode is not None:
            self.top.deiconify()  # Show the settings window again
            self.save_model()
        else:
            self.top.after(100, self.check_training, process)

    def save_model(self):
        input_type_flag = "VEC" if self.input_type.get() == 'vector' else "POS"
        model_name = f"L{self.input_size.get()}_{self.output_size.get()}_{'-'.join([hl.get() for hl in self.hidden_layers if hl.get() != 'none'])}_{self.description.get()}_{'N' if self.normalize.get() else 'U'}_{input_type_flag}.pth"
        messagebox.showinfo("Training Complete", f"Model saved as {model_name}")

        if messagebox.askyesno("Continue", "Do you want to train another model?"):
            if messagebox.askyesno("Keep Settings", "Do you want to keep the current settings?"):
                self.top.destroy()
                self.__init__(self.master)
            else:
                self.top.destroy()
                TrainWindow(self.master)
        else:
            self.close_window()

    def close_window(self):
        self.top.destroy()

if __name__ == "__main__":
    ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"

    root = ctk.CTk()
    root.geometry("500x300")  # Set initial size of the main window
    main_window = MainWindow(root)
    root.mainloop()

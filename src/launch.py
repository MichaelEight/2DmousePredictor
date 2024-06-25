import os
import subprocess
import shutil
import customtkinter as ctk
from tkinter import messagebox, simpledialog
import validate_folders_scheme as vfs
from validate_folders_scheme import folders as vfs_folders

vfs.ensure_folders_exist(vfs_folders)

# Helper functions
def get_pth_files(folder):
    return [f for f in os.listdir(folder) if f.endswith('.pth')]

def get_txt_files(folder):
    return [f for f in os.listdir(folder) if f.endswith('.txt')]

def count_lines(filepath):
    with open(os.path.join('data/data_queue', filepath), 'r') as file:
        return len(file.readlines())

def launch_main(selected_models, selected_classifier):
    args = ["python", "src/main.py", "--models"] + selected_models
    if selected_classifier:
        args += ["--classifier", selected_classifier]
    subprocess.run(args)

def launch_train_predictor(args):
    subprocess.run(["python", "src/train_predictor.py"] + args)

def launch_train_classifier(args):
    subprocess.run(["python", "src/train_classifier.py"] + args)
    
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

    def start_simulation(self):
        self.simulation_window = SimulationWindow(self.master)

    def train(self):
        self.train_window = TrainWindow(self.master)

    def data_viewer(self):
        subprocess.run(["python", "src/mouse_data_viewer.py"])

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

        self.all_files = get_pth_files('models/trained_models')
        self.classifier_files = [f for f in self.all_files if f.startswith('classifier')]
        self.predictor_files = [f for f in self.all_files if not f.startswith('classifier')]

        ctk.CTkLabel(self.top, text="Select Predictor Models:", font=ctk.CTkFont(size=18)).pack(pady=10)
        self.predictor_vars = {}
        for file in self.predictor_files:
            var = ctk.StringVar(value="0")
            chk = ctk.CTkCheckBox(self.top, text=file, variable=var, onvalue="1", offvalue="0")
            chk.pack(anchor='w', padx=20)
            self.predictor_vars[file] = var

        ctk.CTkLabel(self.top, text="Select Classifier Model:", font=ctk.CTkFont(size=18)).pack(pady=10)
        self.classifier_var = ctk.StringVar(value="None")
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
        selected_predictors = [file for file, var in self.predictor_vars.items() if var.get() == "1"]
        selected_classifier = self.classifier_var.get() if self.classifier_var.get() != "None" else None

        launch_main(selected_predictors, selected_classifier)
        self.ask_save_data()

    def ask_save_data(self):
        if messagebox.askyesno("Save Data", "Do you want to save mouse data?"):
            new_name = simpledialog.askstring("Input", "Enter new file name:", parent=self.top)
            if new_name:
                shutil.move('data/data_mouse/mouse_positions.txt', f'data/data_queue/{new_name}.txt')
        self.top.destroy()

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

        self.data_files = get_txt_files('data/data_queue')  # Corrected path

        ctk.CTkLabel(self.top, text="Select Data Files:", font=ctk.CTkFont(size=18)).pack(pady=10)
        self.data_vars = {}
        for file in self.data_files:
            var = ctk.StringVar()
            chk = ctk.CTkCheckBox(self.top, text=f"{file} ({count_lines(file)})", variable=var)
            chk.pack(anchor='w', padx=20)
            self.data_vars[file] = var

        ctk.CTkLabel(self.top, text="Training Type:", font=ctk.CTkFont(size=18)).pack(pady=10)
        self.train_type = ctk.StringVar(value="predictor")
        ctk.CTkRadioButton(self.top, text="Predictor", variable=self.train_type, value="predictor").pack(anchor='w', padx=20)
        ctk.CTkRadioButton(self.top, text="Classifier", variable=self.train_type, value="classifier").pack(anchor='w', padx=20)

        ctk.CTkLabel(self.top, text="Hidden Layers:", font=ctk.CTkFont(size=18)).pack(pady=10)
        self.hidden_layers = []
        for i in range(3):
            var = ctk.StringVar(value="none")
            self.hidden_layers.append(var)
            options = ["none", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"]
            dropdown = ctk.CTkComboBox(self.top, values=options, variable=var)
            dropdown.pack(padx=20, pady=5)

        ctk.CTkLabel(self.top, text="Input Size:", font=ctk.CTkFont(size=18)).pack(pady=10)
        self.input_size = ctk.CTkEntry(self.top, width=200)
        self.input_size.pack(pady=5)

        ctk.CTkLabel(self.top, text="Output Size:", font=ctk.CTkFont(size=18)).pack(pady=10)
        self.output_size = ctk.CTkEntry(self.top, width=200)
        self.output_size.pack(pady=5)

        ctk.CTkLabel(self.top, text="Description:", font=ctk.CTkFont(size=18)).pack(pady=10)
        self.description = ctk.CTkEntry(self.top, width=200)
        self.description.pack(pady=5)

        self.normalize = ctk.StringVar()
        ctk.CTkCheckBox(self.top, text="Normalize", variable=self.normalize).pack(pady=10)

        self.train_button = ctk.CTkButton(self.top, text="Train", command=self.train_model)
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

    def train_model(self):
        selected_data = [file for file, var in self.data_vars.items() if var.get() == "1"]
        hidden_layers_str = '-'.join([f"{size.get()}ReLU" for size in self.hidden_layers if size.get() != "none"])

        args = [
            '--sequence_length', self.input_size.get(),
            '--hidden_layers', hidden_layers_str,
            '--desc', self.description.get(),
        ]
        if self.normalize.get():
            args.append('--normalize')

        if self.train_type.get() == 'predictor':
            args.extend(['--output_size', self.output_size.get()])
            launch_train_predictor(args)
        else:
            launch_train_classifier(args)

        model_name = f"L{self.input_size.get()}_{self.output_size.get()}_{hidden_layers_str}_{self.description.get()}_{'N' if self.normalize.get() else 'U'}.pth"
        messagebox.showinfo("Training Complete", f"Model saved as {model_name}")

        if messagebox.askyesno("Continue", "Do you want to train another model?"):
            if messagebox.askyesno("Keep Settings", "Do you want to keep the current settings?"):
                self.top.destroy()
                self.__init__(self.master)
            else:
                self.top.destroy()
                TrainWindow(self.master)
        else:
            self.top.destroy()

if __name__ == "__main__":
    ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"

    root = ctk.CTk()
    root.geometry("500x300")  # Set initial size of the main window
    main_window = MainWindow(root)
    root.mainloop()

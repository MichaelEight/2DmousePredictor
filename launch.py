import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import subprocess
import shutil

# Helper functions
def move_files(selected_files, src_folder, dst_folder):
    for file in selected_files:
        src_path = os.path.join(src_folder, file)
        dst_path = os.path.join(dst_folder, file)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)

def get_pth_files(folder):
    return [f for f in os.listdir(folder) if f.endswith('.pth')]

def get_txt_files(folder):
    return [f for f in os.listdir(folder) if f.endswith('.txt')]

def count_lines(filepath):
    with open(filepath, 'r') as file:
        return len(file.readlines())

def launch_main():
    subprocess.run(["python", "main.py"])

def launch_train_predictor(args):
    subprocess.run(["python", "train_predictor.py"] + args)

def launch_train_classifier(args):
    subprocess.run(["python", "train_classifier.py"] + args)

# Main window
class MainWindow:
    def __init__(self, master):
        self.master = master
        master.title("Main Menu")

        self.start_simulation_button = tk.Button(master, text="Start Simulation", command=self.start_simulation)
        self.start_simulation_button.pack(pady=10)

        self.train_button = tk.Button(master, text="Training", command=self.train)
        self.train_button.pack(pady=10)

        self.data_viewer_button = tk.Button(master, text="Data Viewer", command=self.data_viewer)
        self.data_viewer_button.pack(pady=10)

    def start_simulation(self):
        self.simulation_window = SimulationWindow(self.master)

    def train(self):
        self.train_window = TrainWindow(self.master)

    def data_viewer(self):
        subprocess.run(["python", "mouse_data_viewer.py"])

class SimulationWindow:
    def __init__(self, master):
        self.top = tk.Toplevel(master)
        self.top.title("Start Simulation")
        self.top.geometry("400x600")  # Set size

        # Center the window
        self.center_window(self.top)

        trained_files = get_pth_files('trained_models')
        loaded_files = get_pth_files('models_to_load')

        self.all_files = list(set(trained_files + loaded_files))
        self.classifier_files = [f for f in self.all_files if f.startswith('classifier')]
        self.predictor_files = [f for f in self.all_files if not f.startswith('classifier')]

        tk.Label(self.top, text="Select Predictor Models:").pack(pady=5)
        self.predictor_vars = {}
        for file in self.predictor_files:
            var = tk.BooleanVar(value=file in loaded_files)
            chk = tk.Checkbutton(self.top, text=file, variable=var)
            chk.pack(anchor='w')
            self.predictor_vars[file] = var

        tk.Label(self.top, text="Select Classifier Model:").pack(pady=5)
        self.classifier_var = tk.StringVar(value="None")
        tk.Radiobutton(self.top, text="None", variable=self.classifier_var, value="None").pack(anchor='w')
        for file in self.classifier_files:
            checked = file in loaded_files
            tk.Radiobutton(self.top, text=file, variable=self.classifier_var, value=file).pack(anchor='w')
            if checked:
                self.classifier_var.set(file)

        self.start_button = tk.Button(self.top, text="Start", command=self.start_simulation)
        self.start_button.pack(pady=10)

    def center_window(self, window):
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry(f'{width}x{height}+{x}+{y}')

    def start_simulation(self):
        selected_predictors = [file for file, var in self.predictor_vars.items() if var.get()]
        selected_classifier = self.classifier_var.get()

        move_files(selected_predictors, 'trained_models', 'models_to_load')
        if selected_classifier != "None":
            move_files([selected_classifier], 'trained_models', 'models_to_load')

        for file in os.listdir('models_to_load'):
            if file not in selected_predictors and file != selected_classifier:
                move_files([file], 'models_to_load', 'trained_models')

        launch_main()
        self.ask_save_data()

    def ask_save_data(self):
        if messagebox.askyesno("Save Data", "Do you want to save mouse data?"):
            new_name = simpledialog.askstring("Input", "Enter new file name:", parent=self.top)
            if new_name:
                shutil.move('data/mouse_positions.txt', f'data_queue/{new_name}.txt')
        self.top.destroy()

class TrainWindow:
    def __init__(self, master):
        self.top = tk.Toplevel(master)
        self.top.title("Training")
        self.top.geometry("500x600")  # Set size

        # Center the window
        self.center_window(self.top)

        self.data_files = get_txt_files('data_queue')

        tk.Label(self.top, text="Select Data Files:").pack(pady=5)
        self.data_vars = {}
        for file in self.data_files:
            var = tk.BooleanVar()
            chk = tk.Checkbutton(self.top, text=f"{file} ({count_lines(f'data_queue/{file}')})", variable=var)
            chk.pack(anchor='w')
            self.data_vars[file] = var

        tk.Label(self.top, text="Training Type:").pack(pady=5)
        self.train_type = tk.StringVar(value="predictor")
        tk.Radiobutton(self.top, text="Predictor", variable=self.train_type, value="predictor").pack(anchor='w')
        tk.Radiobutton(self.top, text="Classifier", variable=self.train_type, value="classifier").pack(anchor='w')

        tk.Label(self.top, text="Hidden Layers:").pack(pady=5)
        self.hidden_layers = []
        for i in range(3):
            var = tk.StringVar(value="none")
            self.hidden_layers.append(var)
            options = ["none", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"]
            dropdown = tk.OptionMenu(self.top, var, *options)
            dropdown.pack()

        tk.Label(self.top, text="Input Size:").pack(pady=5)
        self.input_size = tk.Spinbox(self.top, from_=10, to=200)
        self.input_size.pack()

        tk.Label(self.top, text="Output Size:").pack(pady=5)
        self.output_size = tk.Spinbox(self.top, from_=1, to=100)
        self.output_size.pack()

        tk.Label(self.top, text="Description:").pack(pady=5)
        self.description = tk.Entry(self.top, width=20)
        self.description.pack()

        self.normalize = tk.BooleanVar()
        tk.Checkbutton(self.top, text="Normalize", variable=self.normalize).pack()

        self.train_button = tk.Button(self.top, text="Train", command=self.train_model)
        self.train_button.pack(pady=10)

    def center_window(self, window):
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry(f'{width}x{height}+{x}+{y}')

    def train_model(self):
        selected_data = [file for file, var in self.data_vars.items() if var.get()]
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
    root = tk.Tk()
    root.geometry("400x200")  # Set initial size of the main window
    main_window = MainWindow(root)
    root.mainloop()

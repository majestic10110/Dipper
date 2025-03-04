import tkinter as tk
from tkinter import ttk, Menu, messagebox, simpledialog
import pyaudio
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import butter, lfilter
import threading
import time
import matplotlib
import crcmod
import os
from reedsolo import RSCodec
import logging
import random
import string
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import queue
from sklearn.neural_network import MLPClassifier
import pickle
from collections import Counter

# Setup logging
logging.basicConfig(filename='dipper_receive_v1.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Training data log files for each mode
TRAINING_LOG_FILES = {
    "normal": os.path.join(os.path.dirname(__file__), "ai_training_data_normal.log"),
    "robust": os.path.join(os.path.dirname(__file__), "ai_training_data_robust.log"),
    "robust_plus": os.path.join(os.path.dirname(__file__), "ai_training_data_robust_plus.log")
}

# Shared constants
SAMPLE_RATE = 44100
CHUNK = 2048
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "receive_settings_v1.txt")
AI_MODEL_FILE = os.path.join(os.path.dirname(__file__), "dipper_ai_model.pkl")

# Updated tones for 300-2700 Hz (2400 Hz bandwidth)
CHAR_SOUNDS = {
    "A": [(300, 300, "tone")], "B": [(600, 600, "trill")], "C": [(900, 300, "slide")],
    "D": [(1200, 1200, "tone")], "E": [(1500, 1500, "tone")], "F": [(1800, 2100, "slide")],
    "G": [(300, 300, "trill")], "H": [(2100, 2100, "tone")], "I": [(2400, 2400, "tone")],
    "J": [(600, 900, "slide")], "K": [(900, 900, "tone")], "L": [(1200, 1200, "trill")],
    "M": [(1500, 1800, "slide")], "N": [(1800, 1200, "slide")], "O": [(2100, 2100, "tone")],
    "P": [(2400, 2400, "trill")], "Q": [(2700, 2700, "tone")], "R": [(300, 300, "trill")],
    "S": [(600, 900, "slide")], "T": [(900, 1200, "slide")], "U": [(1200, 1200, "tone")],
    "V": [(1500, 1500, "trill")], "W": [(1800, 2100, "slide")], "X": [(2100, 2100, "trill")],
    "Y": [(2400, 1800, "slide")], "Z": [(2700, 2700, "trill")],
    "0": [(300, 600, "slide")], "1": [(600, 900, "slide")], "2": [(900, 1200, "slide")],
    "3": [(1200, 1500, "slide")], "4": [(1500, 1800, "slide")], "5": [(1800, 2100, "slide")],
    "6": [(2100, 2400, "slide")], "7": [(2400, 2700, "slide")], "8": [(2700, 2400, "slide")],
    "9": [(300, 600, "slide")], "!": [(600, 900, "slide")], "/": [(900, 1200, "slide")],
    "-": [(1200, 1200, "trill")], ".": [(1500, 1500, "tone")], " ": [(300, 300, "tone")],
    "@": [(1800, 2100, "slide")]
}

WORD_SOUNDS = {
    "CQ": [(300, 600, "slide")], "DE": [(900, 1200, "slide")],
}

SYMBOL_MAP = {**{char: i for i, char in enumerate(CHAR_SOUNDS.keys())},
              **{word: i + len(CHAR_SOUNDS) for i, word in enumerate(WORD_SOUNDS.keys())},
              "Robust_Preamble": len(CHAR_SOUNDS) + len(WORD_SOUNDS),
              "Robust+_Preamble": len(CHAR_SOUNDS) + len(WORD_SOUNDS) + 1}
REVERSE_MAP = {i: char_or_word for char_or_word, i in SYMBOL_MAP.items()}
V4_SYMBOLS = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
              8: "I", 9: "K", 10: "L", 11: "M", 12: "N", 13: "O", 14: "P", 15: "Q"}
V4_REVERSE_MAP = {v: k for k, v in V4_SYMBOLS.items()}
crc16 = crcmod.mkCrcFun(0x11021, initCrc=0, xorOut=0xFFFF)
RS_CODEC = RSCodec(8)

# End of part 1
class DipperReceiveV1:
    def __init__(self, root):
        self.root = root
        self.root.title("Dipper Receive V1.1 AI")
        self.root.geometry("1000x800")  # Increased height from 700 to 800

        self.running = True
        self.p = pyaudio.PyAudio()
        self.stream_in = None
        self.stream_out = None
        self.rx_thread = None
        self.training_thread = None
        self.input_devices = {}
        self.output_devices = {}
        self.input_device = tk.StringVar(value="Default")
        self.output_device = tk.StringVar(value="Default")
        self.input_volume = tk.DoubleVar(value=50.0)
        self.sensitivity = tk.DoubleVar(value=50.0)
        self.signal_strength = tk.DoubleVar(value=0.0)
        self.filter_var = tk.StringVar(value="none")
        self.colormap = tk.StringVar(value="viridis")
        self.current_mode = tk.StringVar(value="normal")
        self.text_queue = queue.Queue()
        self.spectrum_data = queue.Queue()
        self.settings = self.load_settings()
        self.text_lines = []
        self.ai_enabled = tk.BooleanVar(value=False)
        self.ai_training_data = []
        self.ai_training_labels = []
        self.ai_model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=2000, learning_rate_init=0.01, early_stopping=False)
        self.learning_active = False
        self.training_active = False
        self.learning_thread = None
        self.sent_text = {"normal": [], "robust": [], "robust_plus": []}
        self.decoded_text = {"normal": [], "robust": [], "robust_plus": []}
        self.accuracy = {"normal": 0.0, "robust": 0.0, "robust_plus": 0.0}
        self.training_sent = []
        self.training_decoded = []
        self.training_accuracy = 0.0
        self.accuracy_labels = {}
        self.last_spectrum = np.zeros(CHUNK//2)
        self.mute_audio = tk.BooleanVar(value=False)
        self.offline_mode = tk.BooleanVar(value=False)
        self.initial_gaps = {"normal": 0.2, "robust": 0.2, "robust_plus": 0.2}
        self.gap_map = self.initial_gaps.copy()
        self.last_save_time = 0
        self.incorrect_count = 0
        self.last_sent = None
        self.decode_count = 0
        self.all_classes = list(range(len(SYMBOL_MAP)))
        self.normal_classes = list(range(len(CHAR_SOUNDS)))
        self.bypass_audio = False  # Toggle to bypass audio for testing

        # Pre-generate audio for all sounds with mode-specific durations
        self.precomputed_audio_normal = {}
        self.precomputed_audio_robust = {}
        for char, pattern in CHAR_SOUNDS.items():
            self.precomputed_audio_normal[char] = self.generate_sound(pattern, 0.1)
            self.precomputed_audio_robust[char] = self.generate_sound(pattern, 0.05)
        self.precomputed_audio_robust["Robust_Preamble"] = np.concatenate([self.generate_sound(CHAR_SOUNDS[num], 0.05) for num in "1357924"])
        self.precomputed_audio_robust["Robust+_Preamble"] = np.concatenate([self.generate_sound(CHAR_SOUNDS[num], 0.05) for num in "2468135"])

        # Dummy initial fit with all classes
        if not os.path.exists(AI_MODEL_FILE):
            dummy_features = np.zeros((len(SYMBOL_MAP), 4))
            dummy_labels = self.all_classes
            self.ai_model.partial_fit(dummy_features, dummy_labels, classes=self.all_classes)
            logging.info("Initialized MLPClassifier with all classes")

        if os.path.exists(AI_MODEL_FILE):
            with open(AI_MODEL_FILE, 'rb') as f:
                self.ai_model = pickle.load(f)
                self.ai_enabled.set(True)

        self.setup_gui()
        self.root.after(100, self.start_audio)
        self.root.after(1000, self.update_accuracy_display)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_gui(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        settings_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Audio Settings", command=self.show_audio_settings)
        settings_menu.add_checkbutton(label="Enable AI Decoding", variable=self.ai_enabled)
        settings_menu.add_command(label="Train AI Model", command=self.train_ai_popup)
        settings_menu.add_command(label="Save Training Data", command=self.save_training_data)

        learning_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Learning", menu=learning_menu)
        
        start_learning_menu = Menu(learning_menu, tearoff=0)
        learning_menu.add_cascade(label="Start Learning Session", menu=start_learning_menu)
        start_learning_menu.add_command(label="Normal", command=lambda: self.start_learning_session("normal"))
        start_learning_menu.add_command(label="Robust", command=lambda: self.start_learning_session("robust"))
        start_learning_menu.add_command(label="Robust+", command=lambda: self.start_learning_session("robust_plus"))
        
        learning_menu.add_command(label="Stop Learning Session", command=self.stop_learning_session)
        learning_menu.add_command(label="View Accuracy", command=self.view_accuracy)

        self.status_frame = tk.Frame(self.main_frame)
        self.status_frame.pack(fill="x", pady=5)
        self.learning_indicator = tk.Canvas(self.status_frame, width=20, height=20)
        self.learning_indicator.pack(side="right", padx=5)
        self.learning_indicator.create_rectangle(0, 0, 20, 20, fill="grey", tags="learning_light")
        tk.Label(self.status_frame, text="Learning Running:").pack(side="right")
        tk.Checkbutton(self.status_frame, text="Mute Audio", variable=self.mute_audio).pack(side="right", padx=10)
        tk.Checkbutton(self.status_frame, text="Offline Mode", variable=self.offline_mode).pack(side="right", padx=10)

        self.mode_frame = tk.Frame(self.main_frame)
        self.mode_frame.pack(fill="x", pady=5)
        tk.Label(self.mode_frame, text="Receive Mode:").pack(side="left", padx=5)
        tk.Radiobutton(self.mode_frame, text="Normal", variable=self.current_mode, value="normal").pack(side="left", padx=5)
        tk.Radiobutton(self.mode_frame, text="Robust", variable=self.current_mode, value="robust").pack(side="left", padx=5)
        tk.Radiobutton(self.mode_frame, text="Robust+", variable=self.current_mode, value="robust_plus").pack(side="left", padx=5)

        self.text_frame = tk.Frame(self.main_frame)
        self.text_frame.pack(fill="both", expand=True, pady=5)
        self.text_canvas = tk.Canvas(self.text_frame, height=300, width=780, bg="black")
        self.text_canvas.pack(side="left", fill="both", expand=True)
        self.control_frame_right = tk.Frame(self.text_frame, bg="black")
        self.control_frame_right.pack(side="right", fill="y", padx=5)
        self.clear_button = tk.Button(self.control_frame_right, text="Clear Text", command=self.clear_text)
        self.clear_button.pack(pady=5)
        for mode in ["normal", "robust", "robust_plus"]:
            label = tk.Label(self.control_frame_right, text=f"{mode.capitalize()} Accuracy: 0.0%", fg="white", bg="black")
            label.pack(pady=2)
            self.accuracy_labels[mode] = label

        self.filter_frame = tk.LabelFrame(self.main_frame, text="Tone Bypass Filter")
        self.filter_frame.pack(fill="x", pady=5)
        tk.Radiobutton(self.filter_frame, text="No Filter", variable=self.filter_var, value="none").pack(side="left", padx=5)
        tk.Radiobutton(self.filter_frame, text="300-2700 Hz", variable=self.filter_var, value="300-2700").pack(side="left", padx=5)

        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(fill="x", pady=5)
        tk.Label(self.control_frame, text="Sensitivity (0-100):").pack(side="left", padx=5)
        tk.Scale(self.control_frame, from_=0, to=100, variable=self.sensitivity, orient=tk.HORIZONTAL, 
                 length=200).pack(side="left", padx=5)
        tk.Label(self.control_frame, text="Signal Strength:").pack(side="left", padx=5)
        self.signal_label = tk.Label(self.control_frame, text="0 dB")
        self.signal_label.pack(side="left", padx=5)
        tk.Label(self.control_frame, text="Waterfall Colormap:").pack(side="right", padx=5)
        tk.OptionMenu(self.control_frame, self.colormap, "viridis", "plasma", "inferno", "magma", 
                      command=self.update_colormap).pack(side="right", padx=5)

        self.waterfall_frame = tk.Frame(self.main_frame)
        self.waterfall_frame.pack(fill="both", expand=True, pady=5)
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.waterfall_data = np.zeros((200, int(CHUNK * 1_000_000 / SAMPLE_RATE)))
        self.waterfall = self.ax.imshow(self.waterfall_data, aspect='auto', cmap=self.colormap.get(), 
                                        extent=[-0.5, 0.5, 0, 200], vmin=-60, vmax=0)  # -500 kHz to +500 kHz
        self.ax.set_xlabel("Frequency (MHz)")
        self.ax.set_ylabel("Time (s)")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.waterfall_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

# End of part 2
    def load_settings(self):
        settings = {"input_device": "Default", "input_volume": 50.0, "filter": "none"}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r") as f:
                    for line in f:
                        if line.strip():
                            key, value = line.strip().split("=", 1)
                            if key in settings:
                                settings[key] = value if key != "input_volume" else float(value)
            except Exception as e:
                logging.error(f"Error loading settings: {e}")
        self.input_device = tk.StringVar(value=settings["input_device"])
        self.input_volume.set(settings["input_volume"])
        self.filter_var.set(settings["filter"])
        return settings

    def save_settings(self):
        try:
            with open(SETTINGS_FILE, "w") as f:
                f.write(f"input_device={self.input_device.get()}\n")
                f.write(f"input_volume={self.input_volume.get()}\n")
                f.write(f"filter={self.filter_var.get()}\n")
            logging.info("Settings saved successfully")
        except Exception as e:
            logging.error(f"Error saving settings: {e}")

    def get_audio_devices(self):
        input_devices = {}
        output_devices = {}
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            if dev["maxInputChannels"] > 0:
                input_devices[i] = dev["name"]
            if dev["maxOutputChannels"] > 0:
                output_devices[i] = dev["name"]
        self.input_devices = input_devices
        self.output_devices = output_devices
        logging.debug(f"Input devices: {input_devices}, Output devices: {output_devices}")

    def show_audio_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Audio Settings")
        settings_window.geometry("400x200")
        settings_window.resizable(False, False)

        tk.Label(settings_window, text="Input Device:").pack(pady=5)
        self.get_audio_devices()
        input_options = ["Default"] + list(self.input_devices.values())
        tk.OptionMenu(settings_window, self.input_device, *input_options).pack(pady=5)

        tk.Label(settings_window, text="Volume (%):").pack(pady=5)
        tk.Scale(settings_window, from_=0, to=100, variable=self.input_volume, orient=tk.HORIZONTAL, 
                 length=200).pack(pady=5)

        tk.Button(settings_window, text="Save", command=lambda: [self.save_settings(), self.safe_update_audio_device(), settings_window.destroy()]).pack(pady=10)

    def safe_update_audio_device(self):
        try:
            input_value = self.input_device.get()
            logging.info(f"Updating audio device to: {input_value}")
            if input_value == "Default":
                input_idx = None
            else:
                if not self.input_devices:
                    self.get_audio_devices()
                input_idx = list(self.input_devices.keys())[list(self.input_devices.values()).index(input_value)]

            if self.stream_in:
                self.stream_in.stop_stream()
                self.stream_in.close()
            if self.stream_out:
                self.stream_out.stop_stream()
                self.stream_out.close()
            if self.rx_thread and self.rx_thread.is_alive():
                self.running = False
                self.rx_thread.join(timeout=1.0)
            self.running = True

            logging.info("Opening input stream")
            self.stream_in = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True, 
                                        frames_per_buffer=CHUNK, input_device_index=input_idx)
            logging.info("Opening output stream")
            self.stream_out = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, output=True)
            logging.info("Audio streams initialized successfully")
            self.rx_thread = threading.Thread(target=self.receive_loop, daemon=True)
            self.rx_thread.start()
            self.root.after(50, self.update_waterfall)
            self.root.after(100, self.update_text)
        except Exception as e:
            logging.error(f"Error in update_audio_device: {e}")
            self.running = False
            if self.stream_in:
                self.stream_in.close()
            if self.stream_out:
                self.stream_out.close()

    def start_audio(self):
        try:
            self.get_audio_devices()
            self.safe_update_audio_device()
        except Exception as e:
            logging.error(f"Error starting audio: {e}")
            self.running = False

    def clear_text(self):
        for line_id in self.text_lines:
            self.text_canvas.delete(line_id)
        self.text_lines = []
        logging.info("Text field cleared")

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_filter(self, data):
        filter_type = self.filter_var.get()
        if filter_type == "none":
            logging.debug("No filter applied")
            return data
        elif filter_type == "300-2700":
            b, a = self.butter_bandpass(300, 2700, SAMPLE_RATE)
            filtered = lfilter(b, a, data)
            logging.debug(f"Applied 300-2700 Hz filter, data max: {np.max(filtered)}")
            return filtered
        return data

    def generate_sound(self, pattern, duration=0.05):
        audio = np.array([], dtype=np.float32)
        if not pattern:
            return np.zeros(int(SAMPLE_RATE * duration), dtype=np.float32)
        for start_freq, end_freq, sound_type in pattern:
            t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
            if sound_type == "tone":
                signal = np.sin(2 * np.pi * start_freq * t)
            elif sound_type == "slide":
                freqs = np.linspace(start_freq, end_freq, len(t))
                signal = np.sin(2 * np.pi * freqs * t)
            elif sound_type == "trill":
                signal = np.sin(2 * np.pi * start_freq * t) * np.sin(2 * np.pi * 20 * t)
            audio = np.concatenate((audio, signal))
        return audio / np.max(np.abs(audio))

    def train_ai_popup(self):
        logging.info("Opening Train AI Model popup")
        training_window = tk.Toplevel(self.root)
        training_window.title("AI Training")
        training_window.geometry("400x200")
        training_window.resizable(False, False)

        label = tk.Label(training_window, text="Select mode and audio output...")
        label.pack(pady=5)
        char_label = tk.Label(training_window, text="Current Training: ")
        char_label.pack(pady=5)

        mode_frame = tk.Frame(training_window)
        mode_frame.pack(pady=5)
        mode_var = tk.StringVar(value="normal")
        tk.Radiobutton(mode_frame, text="Normal", variable=mode_var, value="normal").pack(side="left", padx=5)
        tk.Radiobutton(mode_frame, text="Robust", variable=mode_var, value="robust").pack(side="left", padx=5)
        tk.Radiobutton(mode_frame, text="Robust+", variable=mode_var, value="robust_plus").pack(side="left", padx=5)

        output_frame = tk.Frame(training_window)
        output_frame.pack(pady=5)
        tk.Label(output_frame, text="Audio Output:").pack(side="left", padx=5)
        self.get_audio_devices()
        output_options = ["Default"] + list(self.output_devices.values())
        tk.OptionMenu(output_frame, self.output_device, *output_options).pack(side="left", padx=5)

        button_frame = tk.Frame(training_window)
        button_frame.pack(pady=10)
        start_button = tk.Button(button_frame, text="Start Training", command=lambda: self.start_training(char_label, mode_var))
        start_button.pack(side="left", padx=5)
        stop_button = tk.Button(button_frame, text="Stop Training", command=lambda: self.stop_training(training_window, mode_var))
        stop_button.pack(side="left", padx=5)

    def start_training(self, char_label, mode_var):
        if not self.training_active:
            logging.info(f"Start Training button clicked for mode: {mode_var.get()}")
            print(f"Starting training for {mode_var.get()}")
            self.training_active = True
            self.last_save_time = time.time()
            if not self.stream_in or not self.stream_out:
                self.safe_update_audio_device()
                logging.info("Forced audio initialization for training")
            try:
                self.training_thread = threading.Thread(target=self.train_cycle, args=(char_label, mode_var), daemon=True)
                self.training_thread.start()
                logging.info("Training thread started")
                print("Training thread launched")
            except Exception as e:
                logging.error(f"Failed to start training thread: {e}")
                print(f"Thread start failed: {e}")
                self.training_active = False

    def train_cycle(self, char_label, mode_var):
        logging.info("Thread entered train_cycle")
        print("Inside train_cycle")
        self.ai_training_data = []
        self.ai_training_labels = []
        self.training_sent = []
        self.training_decoded = []
        mode = mode_var.get()
        duration_map = {"normal": 0.1, "robust": 0.05, "robust_plus": 0.05}
        normal_gap = {"normal": 0.05, "robust": 0.025, "robust_plus": 0.025}
        training_gap = 0.2
        logging.info(f"Starting {mode} training (continues until stopped)")

        output_value = self.output_device.get()
        output_idx = None if output_value == "Default" else list(self.output_devices.keys())[list(self.output_devices.values()).index(output_value)]
        training_stream_out = None
        if not self.bypass_audio:
            try:
                logging.info(f"Attempting to open training output stream: {output_value}")
                training_stream_out = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, output=True, 
                                                 output_device_index=output_idx)
                logging.info(f"Training audio output opened on device: {output_value}")
            except Exception as e:
                logging.error(f"Failed to open training audio output: {e}")
                self.training_active = False
                return

        sample_count = 0
        while self.training_active and self.running:
            try:
                item = random.choice(list(CHAR_SOUNDS.keys())) if mode == "normal" else random.choice(list(CHAR_SOUNDS.keys()) + ["Robust_Preamble", "Robust+_Preamble"])
                char_label.config(text=f"Current Training: {item}")
                label = SYMBOL_MAP[item]
                item_type = "preamble" if "Preamble" in item else "character"
                duration = duration_map[mode] * (7 if item_type == "preamble" else 1)
                
                audio = (self.precomputed_audio_normal if mode == "normal" else self.precomputed_audio_robust)[item]
                if not self.bypass_audio and training_stream_out:
                    logging.info("Writing to training_stream_out")
                    training_stream_out.write(audio.astype(np.float32).tobytes())
                    logging.info("Write completed")
                
                if not self.bypass_audio:
                    logging.info("Reading from stream_in")
                    data = self.stream_in.read(int(SAMPLE_RATE * duration), exception_on_overflow=False)
                    logging.info(f"Read completed, data length: {len(data)}")
                    if not data or len(data) == 0:
                        logging.warning("Empty audio data received, retrying...")
                        time.sleep(0.1)
                        continue
                    data = np.frombuffer(data, dtype=np.float32)
                else:
                    data = audio  # Use generated audio as input
                    logging.info("Bypassing audio I/O with simulated data")

                if not np.all(np.isfinite(data)):
                    logging.warning(f"Invalid audio data detected: {data}")
                    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
                data = np.clip(data, -1.0, 1.0) * (self.input_volume.get() / 5.0)
                logging.debug(f"Raw audio data min: {np.min(data)}, max: {np.max(data)}")
                filtered_data = self.apply_filter(data)
                spectrum = np.abs(fft(filtered_data))[:CHUNK//2]
                features = self.extract_features(spectrum[:100])
                self.ai_training_data.append(features)
                self.ai_training_labels.append(label)
                self.training_sent.append(REVERSE_MAP[label])
                
                predicted = self.decode_audio(filtered_data)
                decoded_item = REVERSE_MAP.get(predicted, " ") if predicted is not None else " "
                self.training_decoded.append(decoded_item)
                
                sample_count += 1
                if sample_count % 10 == 0:
                    self.ai_model.partial_fit(self.ai_training_data[-10:], self.ai_training_labels[-10:], 
                                              classes=self.all_classes)
                    logging.info(f"Incremental training after {sample_count} samples")

                if len(self.training_sent) > 0:
                    correct = sum(1 for s, d in zip(self.training_sent, self.training_decoded) if s == d)
                    total = len(self.training_sent)
                    self.training_accuracy = (correct / total) * 100
                    logging.info(f"Training {mode} accuracy: {self.training_accuracy:.2f}%")

                training_gap = max(normal_gap[mode], 0.2 - (self.training_accuracy / 60.0) * (0.2 - normal_gap[mode]))
                logging.info(f"Training gap adjusted to: {training_gap:.3f}s")
                logging.info(f"Trained {mode} {item_type}: {REVERSE_MAP[label]}, peak freq: {np.argmax(spectrum)}")
                self.root.update()
            except Exception as e:
                logging.error(f"Training loop error: {e}")
                time.sleep(0.1)
                continue

            current_time = time.time()
            if current_time - self.last_save_time >= 300:
                self.save_training_data_to_log(mode)
                self.last_save_time = current_time
            
            time.sleep(training_gap)
        
        if training_stream_out:
            training_stream_out.stop_stream()
            training_stream_out.close()
            logging.info("Training audio output closed")
        if self.running:
            self.train_ai()
            self.save_training_data_to_log(mode)
        logging.info("Exiting train_cycle")

    def stop_training(self, window, mode_var):
        self.training_active = False
        mode = mode_var.get()
        self.train_ai()
        self.save_training_data_to_log(mode)
        window.destroy()

    def save_training_data_to_log(self, mode):
        try:
            log_file = TRAINING_LOG_FILES[mode]
            with open(log_file, 'a') as f:
                f.write(f"\n--- Training Session: {time.ctime()} (Mode: {mode}) ---\n")
                for features, label in zip(self.ai_training_data, self.ai_training_labels):
                    char = REVERSE_MAP[label]
                    f.write(f"Character: {char}, Features: {list(features)}\n")
            logging.info(f"Training data saved to {log_file}")
        except Exception as e:
            logging.error(f"Error saving training data to log for {mode}: {e}")

    def train_ai(self):
        if not self.ai_training_data or not self.ai_training_labels:
            logging.warning("No training data available")
            messagebox.showwarning("Training", "No training data collected yet.")
            return
        
        label_counts = Counter(self.ai_training_labels)
        logging.info(f"Label counts: {label_counts}")
        
        if min(label_counts.values()) < 2:
            logging.warning("Some classes have fewer than 2 samples, may affect training")
        
        self.ai_model.fit(self.ai_training_data, self.ai_training_labels)
        
        logging.info(f"AI model fully trained with {self.ai_model.n_iter_} iterations")
        if self.ai_model.n_iter_ == 2000:
            logging.warning("Maximum iterations (2000) reached without convergence")
        else:
            logging.info("Model converged successfully")
        
        with open(AI_MODEL_FILE, 'wb') as f:
            pickle.dump(self.ai_model, f)
        logging.info("AI model trained and saved")
        self.ai_enabled.set(True)

# End of part 3
    def start_learning_session(self, mode):
        if not self.learning_active:
            logging.info(f"Start Learning Session triggered for mode: {mode}")
            print(f"Starting learning session for {mode}")
            self.learning_active = True
            self.current_mode.set(mode)
            self.sent_text = {"normal": [], "robust": [], "robust_plus": []}
            self.decoded_text = {"normal": [], "robust": [], "robust_plus": []}
            self.gap_map = self.initial_gaps.copy()
            self.incorrect_count = 0
            self.last_sent = None
            self.decode_count = 0
            if not self.stream_out or not self.stream_in:
                self.safe_update_audio_device()
                logging.info("Forced audio initialization for learning")
            try:
                self.learning_thread = threading.Thread(target=self.simulate_transmission, args=(mode,), daemon=True)
                self.learning_thread.start()
                logging.info(f"Learning thread started for {mode}")
                print(f"Learning thread launched for {mode}")
            except Exception as e:
                logging.error(f"Failed to start learning thread: {e}")
                print(f"Thread start failed: {e}")
                self.learning_active = False

    def stop_learning_session(self):
        self.learning_active = False
        if self.learning_thread:
            self.learning_thread.join(timeout=1.0)
        self.calculate_accuracy()
        self.train_ai()
        logging.info("Learning session stopped")

    def view_accuracy(self):
        message = (f"Normal Mode Accuracy: {self.accuracy['normal']:.2f}%\n"
                   f"Robust Mode Accuracy: {self.accuracy['robust']:.2f}%\n"
                   f"Robust+ Mode Accuracy: {self.accuracy['robust_plus']:.2f}%")
        messagebox.showinfo("Learning Accuracy", message)

    def update_accuracy_display(self):
        for mode in ["normal", "robust", "robust_plus"]:
            self.accuracy_labels[mode].config(text=f"{mode.capitalize()} Accuracy: {self.accuracy[mode]:.2f}%")
        self.root.after(1000, self.update_accuracy_display)

    def simulate_transmission(self, mode):
        logging.info("Thread entered simulate_transmission")
        print("Inside simulate_transmission")
        self.learning_indicator.itemconfig("learning_light", fill="green")
        mode_duration = 600
        start_time = time.time()
        logging.info(f"Simulating {mode} mode for 10 minutes with gap: {self.gap_map[mode]}")
        
        while self.learning_active and (time.time() - start_time < mode_duration):
            try:
                if mode == "normal":
                    char = random.choice(list(CHAR_SOUNDS.keys()))
                    audio = self.precomputed_audio_normal[char]
                    self.sent_text["normal"].append(char)
                    self.text_queue.put(f"Sent (Normal): {char}")
                    self.last_sent = (char, time.time())
                    self.decode_count = 0  # Reset for one decode per send
                    if not self.bypass_audio and not self.mute_audio.get() and self.stream_out:
                        logging.info("Writing to stream_out")
                        self.stream_out.write(audio.astype(np.float32).tobytes())
                        logging.info("Write completed")
                    if self.offline_mode.get():
                        self.spectrum_data.put(np.abs(fft(audio)[:CHUNK//2]))
                    
                    # Wait for exactly one decode before next send
                    decode_timeout = time.time() + 1.0  # 1 second timeout
                    while self.decode_count < 1 and time.time() < decode_timeout and self.learning_active:
                        time.sleep(0.01)  # Small delay to allow decode
                    if self.decode_count == 0:
                        logging.warning(f"No decode received for {char} in Normal mode")
                    elif self.decode_count > 1:
                        logging.warning(f"Multiple decodes ({self.decode_count}) for {char}, expected 1")
                    time.sleep(self.gap_map["normal"])
                else:
                    if random.choice([True, False]):
                        char = random.choice(list(CHAR_SOUNDS.keys()))
                        audio = self.precomputed_audio_robust[char]
                        self.sent_text[mode].append(char)
                        self.text_queue.put(f"Sent ({mode}): {char}")
                        self.last_sent = (char, time.time())
                    else:
                        audio = self.precomputed_audio_robust["Robust_Preamble" if mode == "robust" else "Robust+_Preamble"]
                        self.text_queue.put(f"Sent ({mode}): Preamble")
                        self.last_sent = (None, time.time())  # Preamble doesn’t need char tracking
                    
                    self.decode_count = 0  # Reset for one decode per send
                    if not self.bypass_audio and not self.mute_audio.get() and self.stream_out:
                        logging.info("Writing to stream_out")
                        self.stream_out.write(audio.astype(np.float32).tobytes())
                        logging.info("Write completed")
                    if self.offline_mode.get():
                        self.spectrum_data.put(np.abs(fft(audio)[:CHUNK//2]))
                    
                    # Wait for exactly one decode before next send
                    decode_timeout = time.time() + 1.0
                    while self.decode_count < 1 and time.time() < decode_timeout and self.learning_active:
                        time.sleep(0.01)
                    if self.decode_count == 0:
                        logging.warning(f"No decode received in {mode} mode")
                    elif self.decode_count > 1:
                        logging.warning(f"Multiple decodes ({self.decode_count}) in {mode}, expected 1")
                    time.sleep(self.gap_map[mode])

                self.calculate_accuracy()
                self.adjust_learning_speed(mode)
                self.auto_save_training_data()
            except Exception as e:
                logging.error(f"Simulation loop error: {e}")
                time.sleep(0.1)
                continue
        
        self.learning_indicator.itemconfig("learning_light", fill="grey")
        logging.info("Exiting simulate_transmission")

    def adjust_learning_speed(self, mode):
        if self.accuracy[mode] < 50.0 and self.gap_map[mode] < 2.0:
            self.gap_map[mode] *= 1.1
            logging.info(f"Slowed {mode} gap to {self.gap_map[mode]:.3f}s due to accuracy {self.accuracy[mode]:.2f}%")
        elif self.accuracy[mode] >= 60.0 and self.gap_map[mode] > 0.001:
            self.gap_map[mode] *= 0.9
            logging.info(f"Speed increased for {mode}, new gap: {self.gap_map[mode]:.3f}s due to accuracy {self.accuracy[mode]:.2f}%")

    def calculate_accuracy(self):
        for mode in ["normal", "robust", "robust_plus"]:
            sent = self.sent_text[mode]
            decoded = self.decoded_text[mode]
            if not sent:
                self.accuracy[mode] = 0.0
                continue
            correct = sum(1 for s, d in zip(sent, decoded) if s == d)
            total = len(sent)
            self.accuracy[mode] = (correct / total) * 100 if total > 0 else 0.0
            logging.info(f"{mode} accuracy: {self.accuracy[mode]:.2f}%")

    def receive_loop(self):
        buffer = []
        preamble_detected = False
        duration_map = {"normal": 0.1, "robust": 0.05, "robust_plus": 0.01}
        while self.running:
            try:
                mode = self.current_mode.get()
                duration = duration_map[mode]
                chunk_size = int(SAMPLE_RATE * (duration + 0.025))
                if not self.offline_mode.get() and not self.bypass_audio:
                    data = self.stream_in.read(chunk_size, exception_on_overflow=False)
                    if not data:
                        logging.warning("No audio data read")
                        time.sleep(0.1 if mode == "normal" else 0.025)
                        continue
                    data = np.frombuffer(data, dtype=np.float32)
                    if not np.all(np.isfinite(data)):
                        logging.warning(f"Invalid audio data detected: {data}")
                        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
                    data = np.clip(data, -1.0, 1.0) * (self.input_volume.get() / 5.0)
                else:
                    if self.spectrum_data.empty():
                        time.sleep(0.1 if mode == "normal" else 0.025)
                        continue
                    spectrum = self.spectrum_data.get()
                    data = np.real(ifft(np.pad(spectrum, (0, CHUNK//2), 'constant')))
                    data = np.clip(data, -1.0, 1.0) * (self.input_volume.get() / 5.0)

                logging.debug(f"Raw data max: {np.max(np.abs(data))}")
                filtered_data = self.apply_filter(data)
                spectrum = np.abs(fft(filtered_data))[:CHUNK//2]
                logging.debug(f"Spectrum max: {np.max(spectrum)}, shape: {spectrum.shape}")
                if not self.offline_mode.get():
                    self.spectrum_data.put(spectrum)
                self.last_spectrum = spectrum

                if not preamble_detected and len(buffer) == 0:
                    if mode == "robust" and self.detect_v4_preamble(filtered_data[:int(SAMPLE_RATE * 0.4)]):
                        preamble_detected = True
                        logging.info("Robust preamble detected")
                        continue
                    elif mode == "robust_plus" and self.detect_robust_plus_preamble(filtered_data[:int(SAMPLE_RATE * 0.4)]):
                        preamble_detected = True
                        logging.info("Robust+ preamble detected")
                        continue

                symbol = self.decode_audio(filtered_data)
                if symbol is not None:
                    if mode in ["robust", "robust_plus"]:
                        buffer.append(symbol)
                        expected_buffer_size = 31 if mode == "robust" else 30
                        if len(buffer) >= expected_buffer_size:
                            packet_id, decoded_text = (self.decode_v4_packet(buffer) if mode == "robust" 
                                                      else self.decode_ofdm_packet(buffer))
                            if decoded_text and decoded_text.strip():
                                decoded_char = decoded_text[0] if mode == "robust" else decoded_text
                                self.text_queue.put(f"Decoded ({mode}): {decoded_char}")
                                if self.learning_active and self.sent_text[mode] and self.decode_count < 1:
                                    self.decoded_text[mode].append(decoded_char)
                                    if self.last_sent and self.last_sent[0]:  # Check if it’s not a preamble
                                        sent_char, _ = self.last_sent
                                        if decoded_char != sent_char:
                                            self.correct_decode(filtered_data, SYMBOL_MAP[sent_char])
                                        self.decode_count += 1
                            buffer.clear()
                            preamble_detected = False
                    else:
                        decoded_text = self.decode_fec([symbol])
                        if decoded_text:
                            self.text_queue.put(f"Decoded (Normal): {decoded_text}")
                            if self.learning_active and self.sent_text["normal"] and self.decode_count < 1:
                                self.decoded_text["normal"].append(decoded_text)
                                if self.last_sent and self.last_sent[0]:
                                    sent_char, _ = self.last_sent
                                    if decoded_text != sent_char:
                                        self.correct_decode(filtered_data, SYMBOL_MAP[sent_char])
                                    self.decode_count += 1
            except Exception as e:
                logging.error(f"Receive error: {e}")
            time.sleep(0.1 if mode == "normal" else 0.025)

    def correct_decode(self, filtered_data, correct_label):
        if self.ai_enabled.get() and hasattr(self, 'ai_model'):
            features = self.extract_features(np.abs(fft(filtered_data))[:CHUNK//2][:100])
            self.ai_training_data.append(features)
            self.ai_training_labels.append(correct_label)
            self.incorrect_count += 1
            logging.info(f"Added correction for label {REVERSE_MAP[correct_label]}, incorrect count: {self.incorrect_count}")
            if self.incorrect_count >= 5:
                self.ai_model.partial_fit(self.ai_training_data[-5:], self.ai_training_labels[-5:], 
                                         classes=self.all_classes)
                logging.info("Incremental retraining performed")
                self.incorrect_count = 0

    def decode_audio(self, data):
        freqs = np.abs(fft(data)[:len(data)//2])
        freq_axis = np.linspace(0, SAMPLE_RATE // 2, len(freqs))
        peak_idx = np.argmax(freqs)
        peak_freq = freq_axis[peak_idx]
        max_amplitude = np.max(freqs)
        sensitivity_value = self.sensitivity.get() / 100.0
        threshold = 0.001 + (sensitivity_value * 0.05)
        logging.debug(f"Peak freq: {peak_freq}, Max amp: {max_amplitude}, Threshold: {threshold}")
        if max_amplitude < threshold:
            return None
        
        if self.ai_enabled.get() and hasattr(self, 'ai_model'):
            features = self.extract_features(freqs[:100])
            prediction = self.ai_model.predict([features])[0]
            if prediction in REVERSE_MAP:
                if self.current_mode.get() == "normal" and prediction not in self.normal_classes:
                    logging.debug(f"Filtered out non-character prediction {REVERSE_MAP[prediction]} in Normal mode")
                    return None
                logging.debug(f"AI decoded symbol: {REVERSE_MAP[prediction]}")
                if self.learning_active:
                    self.ai_training_data.append(features)
                    self.ai_training_labels.append(prediction)
                return prediction
            logging.debug(f"AI unmatched prediction: {prediction}")
        
        tolerance = 150 if self.current_mode.get() == "normal" else 75
        if not np.isfinite(max_amplitude):
            max_amplitude = 0.0
            logging.warning("Max amplitude was NaN, set to 0.0")
        self.signal_strength.set(20 * np.log10(max_amplitude + 1e-10))
        self.signal_label.config(text=f"{self.signal_strength.get():.1f} dB")
        if self.current_mode.get() in ["robust", "robust_plus"]:
            if self.current_mode.get() == "robust":
                for sym in V4_REVERSE_MAP:
                    if abs(peak_freq - CHAR_SOUNDS[sym][0][0]) < tolerance:
                        logging.debug(f"Rule-based decoded robust symbol: {sym}")
                        return sym
            else:
                tones = 16
                freq_spacing = 150
                for freq_idx in range(tones):
                    expected_freq = 300 + freq_idx * freq_spacing
                    if abs(peak_freq - expected_freq) < tolerance:
                        logging.debug(f"Rule-based decoded robust+ freq: {expected_freq}")
                        return (expected_freq, 0.01)
        else:
            for char, pattern in CHAR_SOUNDS.items():
                if abs(peak_freq - pattern[0][0]) < tolerance:
                    logging.debug(f"Rule-based decoded char: {char}")
                    return SYMBOL_MAP[char]
        logging.debug(f"Unmatched freq: {peak_freq}")
        return None

    def extract_features(self, freqs):
        return np.array([np.mean(freqs), np.max(freqs), np.argmax(freqs), np.var(freqs)])

    def auto_save_training_data(self):
        if self.ai_training_data and self.ai_training_labels:
            logging.info(f"Auto-saving training data: {len(self.ai_training_data)} samples")
            self.train_ai()

    def save_training_data(self):
        symbol = simpledialog.askstring("Training", "Enter symbol to train (e.g., A):")
        if symbol and symbol in SYMBOL_MAP:
            features = self.extract_features(self.last_spectrum[:100])
            self.ai_training_data.append(features)
            self.ai_training_labels.append(SYMBOL_MAP[symbol])
            logging.info(f"Training data saved for symbol: {symbol}")
            messagebox.showinfo("Training", f"Saved training data for {symbol}")

    def encode_v4_packet(self, text, packet_id):
        bits = [int(b) for char in text for b in bin(ord(char))[2:].zfill(8)]
        encoded_bits = self.convolutional_encode(bits)
        interleaved_bits = self.interleave(encoded_bits)
        symbols = []
        for i in range(0, len(interleaved_bits), 4):
            chunk = interleaved_bits[i:i+4]
            value = sum(b << (3-j) for j, b in enumerate(chunk[:4]))
            symbols.append(V4_SYMBOLS.get(value, " "))
        header = [V4_SYMBOLS[packet_id % 16]]
        crc = crc16(text.encode('utf-8'))
        crc_symbols = [V4_SYMBOLS[(crc >> 12) & 0xF], V4_SYMBOLS[(crc >> 8) & 0xF], 
                      V4_SYMBOLS[(crc >> 4) & 0xF], V4_SYMBOLS[crc & 0xF]]
        return header + symbols + crc_symbols

    def encode_ofdm_packet(self, text, packet_id):
        bits = [int(b) for char in text for b in bin(ord(char))[2:].zfill(8)]
        encoded_bits = self.convolutional_encode(bits)
        interleaved_bits = self.interleave(encoded_bits)
        rs_encoded = RSCodec(16).encode(bytes(encoded_bits))
        ofdm_symbols = []
        tones = 16
        symbol_rate = 41.6
        freq_spacing = 150
        for i in range(0, len(rs_encoded), int(symbol_rate)):
            chunk = rs_encoded[i:i+int(symbol_rate)]
            if chunk:
                freq_idx = sum(chunk) % tones
                freq = 300 + freq_idx * freq_spacing
                duration = 0.01
                ofdm_symbols.append((freq, duration))
        crc = crc16(text.encode('utf-8'))
        crc_bits = [int(b) for b in bin(crc)[2:].zfill(16)]
        for i in range(0, len(crc_bits), int(symbol_rate)):
            chunk = crc_bits[i:i+int(symbol_rate)]
            if chunk:
                freq_idx = sum(chunk) % tones
                freq = 300 + freq_idx * freq_spacing
                duration = 0.01
                ofdm_symbols.append((freq, duration))
        return ofdm_symbols

    def generate_ofdm_sound(self, freq, duration):
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
        signal = np.sin(2 * np.pi * freq * t)
        return signal / np.max(np.abs(signal))

    def convolutional_encode(self, bits):
        encoded = []
        state = 0
        for bit in bits:
            encoded.extend([(state ^ bit), bit])
            state = bit
        return encoded

    def interleave(self, bits):
        block_size = 16
        padded = bits + [0] * (block_size - len(bits) % block_size) if len(bits) % block_size else bits
        return [padded[i // 4 + (i % 4) * 4] for i in range(len(padded))]

    def decode_fec(self, symbols):
        return "".join(REVERSE_MAP.get(sym, " ") for sym in symbols).strip()

    def decode_v4_packet(self, symbols):
        if len(symbols) < 5:
            return None, None
        packet_id = V4_REVERSE_MAP.get(symbols[0], 0)
        payload_symbols = symbols[1:-4]
        crc_symbols = symbols[-4:]
        bits = [int(b) for sym in payload_symbols for b in bin(V4_REVERSE_MAP.get(sym, 0))[2:].zfill(4)]
        deinterleaved_bits = self.deinterleave(bits)
        decoded_bits = self.viterbi_decode(deinterleaved_bits)
        text = "".join(chr(sum(b << (7-j) for j, b in enumerate(decoded_bits[i:i+8]))) 
                      for i in range(0, len(decoded_bits), 8) if len(decoded_bits[i:i+8]) == 8)
        crc = crc16(text.encode('utf-8'))
        received_crc = sum(V4_REVERSE_MAP.get(s, 0) << (12 - 4*i) for i, s in enumerate(crc_symbols))
        return packet_id, text if crc == received_crc else None

    def decode_ofdm_packet(self, symbols):
        if len(symbols) < 10:
            return 0, "[ERROR] Insufficient data"
        decoded_bits = []
        tones = 16
        symbol_rate = 41.6
        freq_spacing = 150
        for freq, _ in symbols:
            freq_idx = int((freq - 300) / freq_spacing) % tones
            bits = [int(b) for b in bin(freq_idx)[2:].zfill(int(np.log2(tones)))]
            decoded_bits.extend(bits[:int(symbol_rate)])
        deinterleaved_bits = self.deinterleave(decoded_bits)
        viterbi_decoded = self.viterbi_decode(deinterleaved_bits)
        try:
            rs_decoded = RSCodec(16).decode(bytes(viterbi_decoded))[0]
            text = "".join(chr(sum(b << (7-j) for j, b in enumerate(rs_decoded[i:i+8]))) 
                          for i in range(0, len(rs_decoded), 8) if len(rs_decoded[i:i+8]) == 8)
            crc_pos = len(text) - 2
            if crc_pos <= 0 or not all(c in '0123456789ABCDEFabcdef' for c in text[crc_pos:]):
                return 0, "[ERROR] Invalid CRC"
            crc_received = int(text[crc_pos:], 16)
            crc_calculated = crc16(text[:crc_pos].encode('utf-8'))
            return 0, text if crc_received == crc_calculated else "[ERROR] CRC mismatch"
        except Exception:
            return 0, "[ERROR] Decoding failed"

    def deinterleave(self, bits):
        block_size = 16
        if len(bits) < block_size:
            return bits
        num_blocks = (len(bits) + block_size - 1) // block_size
        padded_length = num_blocks * block_size
        padded_bits = bits + [0] * (padded_length - len(bits))
        deinterleaved = [0] * padded_length
        for i in range(len(padded_bits)):
            deinterleaved[i // 4 + (i % 4) * 4] = padded_bits[i]
        return deinterleaved[:len(bits)]

    def viterbi_decode(self, bits):
        states = {0: (0, []), 1: (float('inf'), [])}
        for i in range(0, len(bits), 2):
            r1, r2 = bits[i], bits[i+1]
            new_states = {}
            for s in [0, 1]:
                for b in [0, 1]:
                    next_s = b
                    o1, o2 = [(0, 0), (1, 1), (1, 0), (0, 1)][s*2 + b]
                    cost = states[s][0] + (r1 ^ o1) + (r2 ^ o2)
                    path = states[s][1] + [b]
                    if next_s not in new_states or cost < new_states[next_s][0]:
                        new_states[next_s] = (cost, path)
            states = new_states
        return min(states.values(), key=lambda x: x[0])[1]

    def detect_v4_preamble(self, data):
        preamble_freqs = [(300, 600), (900, 1200), (1500, 1800), (2100, 2400), 
                         (600, 900), (1200, 1500), (1800, 2100)]
        segment_size = int(SAMPLE_RATE * 0.05)
        freq_axis = np.linspace(0, SAMPLE_RATE // 2, segment_size // 2)
        detected = True
        for i, (start_freq, end_freq) in enumerate(preamble_freqs):
            chunk = data[i * segment_size:(i + 1) * segment_size]
            if len(chunk) < segment_size:
                return False
            freqs = np.abs(fft(chunk)[:segment_size // 2])
            peak_idx = np.argmax(freqs)
            peak_freq = freq_axis[peak_idx]
            if not (start_freq - 200 < peak_freq < end_freq + 200):
                detected = False
                break
        return detected

    def detect_robust_plus_preamble(self, data):
        preamble_freqs = [(300, 600), (900, 1200), (1500, 1800), (2100, 2400), 
                         (600, 900), (1200, 1500), (1800, 2100)]
        segment_size = int(SAMPLE_RATE * 0.05)
        freq_axis = np.linspace(0, SAMPLE_RATE // 2, segment_size // 2)
        detected = True
        for i, (start_freq, end_freq) in enumerate(preamble_freqs):
            chunk = data[i * segment_size:(i + 1) * segment_size]
            if len(chunk) < segment_size:
                return False
            freqs = np.abs(fft(chunk)[:segment_size // 2])
            peak_idx = np.argmax(freqs)
            peak_freq = freq_axis[peak_idx]
            if not (start_freq - 200 < peak_freq < end_freq + 200):
                detected = False
                break
        return detected

    def update_text(self):
        if not self.running:
            return
        try:
            mode = self.current_mode.get()
            while not self.text_queue.empty():
                text = self.text_queue.get_nowait()
                if f"({mode})" in text or (self.learning_active and ("Sent" in text or "Decoded" in text)):
                    formatted_text = text
                    line_height = 20
                    max_lines = 300 // line_height
                    fill_color = "green" if "Decoded" in formatted_text else "white"
                    if len(self.text_lines) >= max_lines:
                        self.text_canvas.delete(self.text_lines.pop(0))
                        for i, line_id in enumerate(self.text_lines):
                            self.text_canvas.coords(line_id, 10, (i + 1) * line_height)
                    new_line_id = self.text_canvas.create_text(10, (len(self.text_lines) + 1) * line_height, 
                                                              text=formatted_text + "\n", anchor="w", fill=fill_color, 
                                                              font=("Courier", 12))
                    self.text_lines.append(new_line_id)
                    logging.info(f"Displayed: {formatted_text} at {time.time():.2f}")
        except Exception as e:
            logging.error(f"Text update error: {e}")
        self.root.after(100, self.update_text)

    def update_colormap(self, *args):
        self.waterfall.set_cmap(self.colormap.get())
        self.canvas.draw()

    def update_waterfall(self):
        if not self.running:
            return
        try:
            if not self.spectrum_data.empty():
                spectrum = self.spectrum_data.get_nowait()
                if len(spectrum) != CHUNK//2:
                    logging.warning(f"Spectrum length mismatch: {len(spectrum)} vs {CHUNK//2}")
                    spectrum = np.pad(spectrum, (0, CHUNK//2 - len(spectrum)), mode='constant')
                raw_dB = 20 * np.log10(np.maximum(spectrum, 1e-10))  # Avoid log(0)
                logging.debug(f"Raw spectrum dB min: {np.min(raw_dB)}, max: {np.max(raw_dB)}")
                
                # Map 0-22 kHz (SAMPLE_RATE/2) to -500 kHz to +500 kHz (1 MHz total width)
                target_bins = CHUNK // 2  # 1024 bins for full spectrum
                freq_range_hz = SAMPLE_RATE / 2  # 22050 Hz
                display_range_hz = 1_000_000  # 1 MHz
                bin_width = freq_range_hz / target_bins  # Hz per bin
                target_bins_display = int(display_range_hz / bin_width)  # Number of bins for 1 MHz
                
                # Center the actual audio data (0-22 kHz) around 0 Hz (-500 kHz to +500 kHz)
                center_bin = target_bins_display // 2  # Middle of the 1 MHz range
                audio_bins = int(freq_range_hz / bin_width)  # ≈ 1024 bins for 22 kHz
                half_audio_bins = audio_bins // 2
                
                # Create a wider array for the 1 MHz display, padding with silence
                spectrum_dB = np.full(target_bins_display, -60, dtype=float)  # Default to -60 dB
                start_idx = center_bin - half_audio_bins
                end_idx = center_bin + half_audio_bins
                spectrum_dB[start_idx:end_idx] = np.clip(raw_dB, -60, 0)  # Place 22 kHz data in center
                
                # Ensure waterfall_data matches the new shape
                if self.waterfall_data.shape[1] != target_bins_display:
                    self.waterfall_data = np.zeros((200, target_bins_display))
                    logging.info(f"Resized waterfall_data to {self.waterfall_data.shape}")
                
                self.waterfall_data = np.roll(self.waterfall_data, -1, axis=0)
                self.waterfall_data[-1, :] = spectrum_dB
                self.waterfall.set_array(self.waterfall_data)
                self.waterfall.set_clim(vmin=-60, vmax=0)
                self.canvas.draw()
                self.canvas.flush_events()  # Force redraw
                logging.debug(f"Waterfall updated, clipped dB min: {np.min(self.waterfall_data[-1, :])}, max: {np.max(self.waterfall_data[-1, :])}")
            else:
                logging.debug("No new spectrum data available")
        except Exception as e:
            logging.error(f"Waterfall update error: {e}")
        self.root.after(50, self.update_waterfall)

    def on_closing(self):
        self.running = False
        self.learning_active = False
        self.training_active = False
        if self.stream_in:
            self.stream_in.stop_stream()
            self.stream_in.close()
        if self.stream_out:
            self.stream_out.stop_stream()
            self.stream_out.close()
        if self.p:
            self.p.terminate()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DipperReceiveV1(root)
    root.mainloop()

# End of part 4
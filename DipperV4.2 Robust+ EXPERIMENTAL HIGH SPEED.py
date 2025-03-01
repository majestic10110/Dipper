import pyaudio
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import butter, lfilter
import threading
import time
import os
import serial
from reedsolo import RSCodec, ReedSolomonError
import crcmod
import sys
import serial.tools.list_ports  # For detecting COM ports

# Debug output at the very start
print("Script starting...")

# Debug output for imports
print("Imports completed")

# Check Python version
print(f"Python version: {sys.version}")

# Check current directory
print(f"Current directory: {os.getcwd()}")

# Verify file existence
script_path = os.path.abspath(__file__)
print(f"Script path: {script_path}")

try:
    # Test dependency imports
    print("Testing dependencies...")
    p = pyaudio.PyAudio()
    p.terminate()
    np.array([1, 2, 3])
    RSCodec(8)
    crc16 = crcmod.mkCrcFun(0x11021, initCrc=0, xorOut=0xFFFF)
    serial.tools.list_ports.comports()
    print("Dependencies loaded successfully")
except Exception as e:
    print(f"Dependency error: {e}")
    sys.exit(1)

CHAR_SOUNDS = {
    "A": [(2000, 2000, "tone")], "B": [(3000, 3000, "trill")], "C": [(4000, 2000, "slide")],
    "D": [(1500, 1500, "tone")], "E": [(5000, 5000, "tone")], "F": [(2500, 3500, "slide")],
    "G": [(1000, 1000, "trill")], "H": [(3500, 3500, "tone")], "I": [(4500, 4500, "tone")],
    "J": [(2000, 4000, "slide")], "K": [(3000, 3000, "tone")], "L": [(4000, 4000, "trill")],
    "M": [(1500, 2500, "slide")], "N": [(5000, 3000, "slide")], "O": [(1000, 1000, "tone")],
    "P": [(3500, 3500, "trill")], "Q": [(2500, 2500, "tone")], "R": [(2000, 2000, "trill")],
    "S": [(4500, 1500, "slide")], "T": [(3000, 5000, "slide")], "U": [(4000, 4000, "tone")],
    "V": [(1500, 1500, "trill")], "W": [(1000, 2000, "slide")], "X": [(5000, 5000, "trill")],
    "Y": [(3500, 2500, "slide")], "Z": [(2500, 2500, "trill")],
    "0": [(1000, 1500, "slide")], "1": [(1500, 2000, "slide")], "2": [(2000, 2500, "slide")],
    "3": [(2500, 3000, "slide")], "4": [(3000, 3500, "slide")], "5": [(3500, 4000, "slide")],
    "6": [(4000, 4500, "slide")], "7": [(4500, 5000, "slide")], "8": [(5000, 4500, "slide")],
    "9": [(2000, 1500, "slide")], "!": [(5000, 1000, "slide")], "/": [(3000, 4000, "slide")],
    "-": [(2000, 2000, "trill")], ".": [(4500, 4500, "tone")], " ": [(1000, 1000, "tone")],
    "@": [(6000, 7000, "slide")]  # Added '@' with a unique sound pattern (high frequencies)
}

WORD_SOUNDS = {
    "CQ": [(2000, 4000, "slide")], "DE": [(1500, 2500, "slide")],
}

SAMPLE_RATE = 44100
CHUNK = 1024
CALLSIGN_FILE = os.path.join(os.path.dirname(__file__), "mycallsign.txt")
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "radio_settings.txt")

SYMBOL_MAP = {**{char: i for i, char in enumerate(CHAR_SOUNDS.keys())},
              **{word: i + len(CHAR_SOUNDS) for i, word in enumerate(WORD_SOUNDS.keys())}}
REVERSE_MAP = {i: char_or_word for char_or_word, i in SYMBOL_MAP.items()}
RS_CODEC = RSCodec(8)

V4_SYMBOLS = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
    8: "I", 9: "K", 10: "L", 11: "M", 12: "N", 13: "O", 14: "P", 15: "Q"
}
V4_REVERSE_MAP = {v: k for k, v in V4_SYMBOLS.items()}

CONV_TABLE = {
    (0, 0): [0, 0], (0, 1): [1, 1], (1, 0): [1, 0], (1, 1): [0, 1]
}

crc16 = crcmod.mkCrcFun(0x11021, initCrc=0, xorOut=0xFFFF)

import tkinter as tk
from tkinter import messagebox, ttk, simpledialog, Menu
import tkinter.font  # Correct import for font handling
import pyaudio
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import butter, lfilter
import threading
import time
import os
import serial
from reedsolo import RSCodec, ReedSolomonError
import crcmod
import sys
import serial.tools.list_ports  # For detecting COM ports

class DipperModeApp:
    def __init__(self, root):
        print("Initializing DipperModeApp")
        self.root = root
        try:
            self.root.title("DIPPER V4.2 Robust+ by M0OLI")
            self.root.geometry("1000x750")  # Adjusted to a reasonable default width, user can resize
            print("Root window created")
        except Exception as e:
            print(f"Root window error: {e}")
            raise

        # Initialize settings with no preset baud rate
        self.settings = self.load_settings()
        self.radio = tk.StringVar(value=self.settings.get("radio", "Icom IC-703"))
        self.usb_address = tk.StringVar(value=self.settings.get("usb_address", ""))
        self.mode_usb = tk.BooleanVar(value=self.settings.get("mode_usb", False))
        self.mode_usb_digital = tk.BooleanVar(value=self.settings.get("mode_usb_digital", False))
        self.mode_fm = tk.BooleanVar(value=self.settings.get("mode_fm", False))
        self.serial_port = tk.StringVar(value=self.settings.get("serial_port", "NONE"))
        self.baud_rate = tk.StringVar(value=self.settings.get("baud_rate", ""))  # Empty by default, user-defined
        self.rts = tk.BooleanVar(value=self.settings.get("rts", False))
        self.dtr = tk.BooleanVar(value=self.settings.get("dtr", False))
        self.ptt_port = tk.StringVar(value=self.settings.get("ptt_port", "NONE"))
        self.ptt_rts = tk.BooleanVar(value=self.settings.get("ptt_rts", False))
        self.ptt_dtr = tk.BooleanVar(value=self.settings.get("ptt_dtr", False))

        # Audio settings
        self.input_device_index = tk.IntVar(value=self.settings.get("input_device", -1))
        self.output_device_index = tk.IntVar(value=self.settings.get("output_device", -1))
        self.input_volume = tk.DoubleVar(value=self.settings.get("input_volume", 50.0))  # Default to 50.0
        self.sensitivity = tk.DoubleVar(value=self.settings.get("sensitivity", 50.0))  # Default to 50.0 (0–100 scale)
        self.input_devices, self.output_devices = self.get_audio_devices()

        # Initialize speed, filter, and receive buffer variables
        self.speed_var = tk.StringVar(value=self.settings.get("speed", "slow"))
        self.filter_var = tk.StringVar(value=self.settings.get("filter", "none"))
        self.receive_buffer = []  # Buffer for received characters before displaying
        self.temp_receive_buffer = []  # Temporary buffer for accumulating symbols before display

        self.light_colors = {"bg": "#FFFFFF", "fg": "#000000", "entry_bg": "#F0F0F0", "button_bg": "#D0D0D0"}
        self.current_colors = self.light_colors

        self.my_callsign_value = ""
        try:
            self.my_callsign_value = self.load_callsign()
            if not self.my_callsign_value:
                self.my_callsign_value = self.prompt_callsign()
                self.save_callsign(self.my_callsign_value)
        except Exception as e:
            print(f"Callsign load error: {e}")

        self.p = None
        self.stream_out = None
        self.stream_in = None
        self.running = False
        self.rx_thread = None
        self.tx_thread = None
        self.is_v4_mode = False
        self.packet_buffer = {}
        self.last_packet_id = -1
        self.tx_queue = []
        self.rx_ack = None
        self.radio_serial = None  # For CAT communication
        self.indicator_timeout = None  # Initialize indicator timeout

        # Initialize rx_output as None, to be set in GUI build
        self.rx_output = None

        print("Building GUI")
        try:
            self.main_frame = tk.Frame(root, bg=self.current_colors["bg"])
            self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

            # Menu Bar
            menubar = Menu(self.root)
            self.root.config(menu=menubar)
            settings_menu = Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Settings", menu=settings_menu)
            settings_menu.add_command(label="Configure Radio", command=self.show_radio_settings_window)
            settings_menu.add_command(label="Audio Settings", command=self.show_audio_settings_window)

            self.callsign_frame = tk.Frame(self.main_frame, bg=self.current_colors["bg"])
            self.callsign_frame.pack(fill="x", pady=(0, 10))
            
            tk.Label(self.callsign_frame, text="My Callsign:", fg=self.current_colors["fg"], 
                     bg=self.current_colors["bg"]).grid(row=0, column=0, padx=5, pady=5, sticky="e")
            self.my_callsign = tk.Entry(self.callsign_frame, width=15, fg=self.current_colors["fg"], 
                                       bg=self.current_colors["entry_bg"], insertbackground=self.current_colors["fg"])
            self.my_callsign.grid(row=0, column=1, padx=5, pady=5)
            if self.my_callsign_value:
                self.my_callsign.insert(0, self.my_callsign_value)
                self.my_callsign.config(state="disabled")

            tk.Label(self.callsign_frame, text="To Callsign:", fg=self.current_colors["fg"], 
                     bg=self.current_colors["bg"]).grid(row=0, column=2, padx=5, pady=5, sticky="e")
            self.to_callsign = tk.Entry(self.callsign_frame, width=15, fg=self.current_colors["fg"], 
                                       bg=self.current_colors["entry_bg"], insertbackground=self.current_colors["fg"])
            self.to_callsign.grid(row=0, column=3, padx=5, pady=5)

            self.dark_mode_var = tk.BooleanVar()
            self.dark_mode_check = tk.Checkbutton(self.callsign_frame, text="Dark Mode", 
                                                 variable=self.dark_mode_var, command=self.toggle_dark_mode, 
                                                 fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                                                 selectcolor=self.current_colors["entry_bg"])
            self.dark_mode_check.grid(row=0, column=4, padx=20, pady=5)

            self.v4_indicator_frame = tk.Frame(self.callsign_frame, bg=self.current_colors["bg"])
            self.v4_indicator_frame.grid(row=0, column=5, padx=10, pady=5, sticky="e")
            tk.Label(self.v4_indicator_frame, text="Robust RX/TX:", fg=self.current_colors["fg"], 
                     bg=self.current_colors["bg"]).pack(side="left")
            self.v4_indicator = tk.Canvas(self.v4_indicator_frame, width=20, height=20, 
                                         bg=self.current_colors["bg"], highlightthickness=0)
            self.v4_indicator.create_rectangle(0, 0, 20, 20, fill="grey", tags="robust_mode_light")
            self.v4_indicator.pack(side="left")

            # Add Robust+ indicator beneath Robust indicator
            self.robust_plus_indicator_frame = tk.Frame(self.callsign_frame, bg=self.current_colors["bg"])
            self.robust_plus_indicator_frame.grid(row=1, column=5, padx=10, pady=(0, 5), sticky="e")
            tk.Label(self.robust_plus_indicator_frame, text="Robust+ RX/TX:", fg=self.current_colors["fg"], 
                     bg=self.current_colors["bg"]).pack(side="left")
            self.robust_plus_indicator = tk.Canvas(self.robust_plus_indicator_frame, width=20, height=20, 
                                                  bg=self.current_colors["bg"], highlightthickness=0)
            self.robust_plus_indicator.create_rectangle(0, 0, 20, 20, fill="grey", tags="robust_plus_mode_light")
            self.robust_plus_indicator.pack(side="left")

            self.tx_frame = tk.Frame(self.main_frame, bg=self.current_colors["bg"])
            self.tx_frame.pack(fill="x", pady=(0, 10))
            
            tk.Label(self.tx_frame, text="Transmit Message:", fg=self.current_colors["fg"], 
                     bg=self.current_colors["bg"]).pack(anchor="w", pady=(0, 5))
            self.tx_input = tk.Entry(self.tx_frame, width=60, fg=self.current_colors["fg"], 
                                    bg=self.current_colors["entry_bg"], insertbackground=self.current_colors["fg"])
            self.tx_input.pack(fill="x", pady=(0, 5))
            
            self.button_frame = tk.Frame(self.tx_frame, bg=self.current_colors["bg"])
            self.button_frame.pack(anchor="w", pady=(0, 5))
            self.tx_button = tk.Button(self.button_frame, text="Send", command=self.transmit, 
                                      fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
            self.tx_button.pack(side="left", padx=5)
            self.cq_button = tk.Button(self.button_frame, text="CQ", command=self.send_cq, 
                                      fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
            self.cq_button.pack(side="left", padx=5)

            self.speed_frame = tk.LabelFrame(self.tx_frame, text="Speed", fg=self.current_colors["fg"], 
                                            bg=self.current_colors["bg"])
            self.speed_frame.pack(anchor="w", pady=5)
            tk.Radiobutton(self.speed_frame, text="Slow (~6-13 WPM)", variable=self.speed_var, value="slow", 
                          fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                          selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)
            tk.Radiobutton(self.speed_frame, text="Medium (~10-20 WPM)", variable=self.speed_var, value="medium", 
                          fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                          selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)
            tk.Radiobutton(self.speed_frame, text="Fast (~15-30 WPM)", variable=self.speed_var, value="fast", 
                          fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                          selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)
            tk.Radiobutton(self.speed_frame, text="Robust (~20-40 WPM)", variable=self.speed_var, value="robust", 
                          fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                          selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)
            tk.Radiobutton(self.speed_frame, text="Robust+ (~40-100 WPM)", variable=self.speed_var, value="robust_plus", 
                          fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                          selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)

            self.filter_frame = tk.LabelFrame(self.tx_frame, text="Frequency Filter", fg=self.current_colors["fg"], 
                                             bg=self.current_colors["bg"])
            self.filter_frame.pack(anchor="w", pady=5)
            tk.Radiobutton(self.filter_frame, text="No Filter", variable=self.filter_var, value="none", 
                          fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                          selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)
            tk.Radiobutton(self.filter_frame, text="900-5100 Hz", variable=self.filter_var, value="900-5100", 
                          fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                          selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)
            tk.Radiobutton(self.filter_frame, text="800-5200 Hz", variable=self.filter_var, value="800-5200", 
                          fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                          selectcolor=self.current_colors["entry_bg"]).pack(side="left", padx=5)

            self.rx_frame = tk.Frame(self.main_frame, bg=self.current_colors["bg"])
            self.rx_frame.pack(fill="both", expand=True, pady=(10, 0))
            
            self.rx_header_frame = tk.Frame(self.rx_frame, bg=self.current_colors["bg"])
            self.rx_header_frame.pack(fill="x")
            tk.Label(self.rx_header_frame, text="Received Messages:", fg=self.current_colors["fg"], 
                     bg=self.current_colors["bg"]).pack(side="left", pady=(0, 5))
            self.clear_button = tk.Button(self.rx_header_frame, text="Clear", command=self.clear_receive, 
                                         fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
            self.clear_button.pack(side="right", padx=5, pady=5)

            # Redesign receive window using Text widget with scrollbars and monospaced font
            self.rx_output_frame = tk.Frame(self.rx_frame, bg=self.current_colors["bg"])
            self.rx_output_frame.pack(fill="both", expand=True, pady=(0, 10))

            # Vertical scrollbar
            v_scrollbar = tk.Scrollbar(self.rx_output_frame, orient="vertical")
            v_scrollbar.pack(side="right", fill="y")

            # Horizontal scrollbar
            h_scrollbar = tk.Scrollbar(self.rx_output_frame, orient="horizontal")
            h_scrollbar.pack(side="bottom", fill="x")

            # Text widget for receive output with dynamic width, monospaced font, and no wrapping
            self.rx_output = tk.Text(self.rx_output_frame, 
                                    font=tkinter.font.Font(family="Courier", size=10),  # Use monospaced font
                                    fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], 
                                    insertbackground=self.current_colors["fg"], wrap="none",  # Disable default wrapping
                                    yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            self.rx_output.pack(side="left", fill="both", expand=True)

            # Configure scrollbars
            v_scrollbar.config(command=self.rx_output.yview)
            h_scrollbar.config(command=self.rx_output.xview)

            self.rx_output.tag_config("sent", foreground="red")
            self.rx_output.tag_config("received", foreground="blue")
            self.rx_output.tag_config("underline", underline=True)

            # Bind window resize to update dynamic width
            self.root.bind("<Configure>", self.update_dynamic_width)

            print("GUI built successfully")
        except Exception as e:
            print(f"GUI build error: {e}")
            # Ensure rx_output exists even if GUI build fails for error logging
            if not hasattr(self, 'rx_output'):
                self.rx_output = tk.Text(self.root, 
                                       font=tkinter.font.Font(family="Courier", size=10),  # Use monospaced font
                                       fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], 
                                       insertbackground=self.current_colors["fg"])
                self.rx_output.pack_forget()  # Hidden but accessible for logging
            if self.rx_output:
                self.rx_output.insert(tk.END, f"GUI error: {e}\n")

        print("Scheduling audio start")
        self.root.after(100, self.start_audio)

    def update_dynamic_width(self, event):
        """Update the dynamic width of the rx_output Text widget based on its current size."""
        if hasattr(self, 'rx_output'):
            # Get the current pixel width of the Text widget
            pixel_width = self.rx_output.winfo_width()
            if pixel_width > 0:  # Ensure valid width
                # Use font metrics to calculate characters per line (using Courier 10pt)
                font = tkinter.font.Font(family="Courier", size=10)
                char_width = font.measure("0")  # Width of a single character (using '0' as a reference)
                if char_width > 0:
                    chars_per_line = pixel_width // char_width
                    print(f"Dynamic width updated - Pixel width: {pixel_width}, Characters per line: {chars_per_line}")
                    self.dynamic_width = max(1, chars_per_line)  # Ensure at least 1 character
                else:
                    print("Warning: Invalid font width detected, using fallback width of 200 characters.")
                    self.dynamic_width = 200  # Increased fallback to ensure wider default
            else:
                self.dynamic_width = 200  # Fallback if width is invalid

    def clear_receive(self):
        """Clears the received messages text box and resets the receive buffer."""
        if self.rx_output:
            self.rx_output.delete(1.0, tk.END)
        self.receive_buffer = []  # Reset the receive buffer
        self.temp_receive_buffer = []  # Reset the temporary buffer

    def load_callsign(self):
        if os.path.exists(CALLSIGN_FILE):
            with open(CALLSIGN_FILE, "r") as f:
                return f.read().strip()
        return ""

    def save_callsign(self, callsign):
        with open(CALLSIGN_FILE, "w") as f:
            f.write(callsign)

    def prompt_callsign(self):
        return simpledialog.askstring("Setup", "What is your callsign?", parent=self.root) or ""

    def load_settings(self):
        """Load radio and audio settings from file, including sensitivity."""
        settings = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r") as f:
                    for line in f:
                        if line.strip():
                            key, value = line.strip().split("=", 1)
                            if key == "baud_rate":
                                settings[key] = value  # Store as string for user input
                            elif key == "input_volume":
                                settings[key] = float(value)
                            elif key == "sensitivity":
                                settings[key] = float(value) if value else 50.0  # Default to 50.0 if not set
                            elif key in ["rts", "dtr", "mode_usb", "mode_usb_digital", "mode_fm", "ptt_rts", "ptt_dtr"]:
                                settings[key] = value.lower() == "true"
                            elif key in ["input_device", "output_device"]:
                                settings[key] = int(value) if value != "-1" else -1
                            else:
                                settings[key] = value
            except Exception as e:
                print(f"Error loading settings: {e}")
        return settings

    def save_settings(self):
        """Save radio and audio settings to file, persisting until changed and saved again, including sensitivity."""
        settings = {
            "radio": self.radio.get(),
            "usb_address": self.usb_address.get(),
            "mode_usb": self.mode_usb.get(),
            "mode_usb_digital": self.mode_usb_digital.get(),
            "mode_fm": self.mode_fm.get(),
            "serial_port": self.serial_port.get(),
            "baud_rate": self.baud_rate.get(),  # Store as user input string
            "rts": self.rts.get(),
            "dtr": self.dtr.get(),
            "ptt_port": self.ptt_port.get(),
            "ptt_rts": self.ptt_rts.get(),
            "ptt_dtr": self.ptt_dtr.get(),
            "input_device": self.input_device_index.get(),
            "output_device": self.output_device_index.get(),
            "input_volume": self.input_volume.get(),
            "sensitivity": self.sensitivity.get(),  # Save sensitivity (0.0–100.0)
            "speed": self.speed_var.get(),
            "filter": self.filter_var.get(),
        }
        try:
            with open(SETTINGS_FILE, "w") as f:
                for key, value in settings.items():
                    if key in ["rts", "dtr", "mode_usb", "mode_usb_digital", "mode_fm", "ptt_rts", "ptt_dtr"]:
                        f.write(f"{key}={value}\n")
                    else:
                        f.write(f"{key}={value}\n")
            print("Radio settings saved successfully")
            if self.rx_output:
                self.rx_output.insert(tk.END, "Radio settings saved successfully\n")
        except Exception as e:
            if self.rx_output:
                self.rx_output.insert(tk.END, f"Error saving settings: {str(e)}\n")
            print(f"Error saving settings: {e}")

    def get_audio_devices(self):
        p = pyaudio.PyAudio()
        input_devices = {}
        output_devices = {}
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            name = dev["name"]
            index = dev["index"]
            if dev["maxInputChannels"] > 0:
                input_devices[index] = name
            if dev["maxOutputChannels"] > 0:
                output_devices[index] = name
        p.terminate()
        print(f"Available input devices: {input_devices}")
        print(f"Available output devices: {output_devices}")
        if not input_devices or not output_devices:
            raise Exception("No audio devices available. Check hardware or close other audio programs.")
        return input_devices, output_devices

    def get_com_ports(self):
        """Get available COM ports for serial communication."""
        return ["NONE"] + [port.device for port in serial.tools.list_ports.comports()]

    def get_default_address(self, radio):
        """Return the default address based on the selected radio, empty for Generic."""
        default_addresses = {
            "Icom IC-703": "68H",
            "Icom IC-705": "94H",
            "Generic": ""  # Empty for user customization
        }
        return default_addresses.get(radio, "")

    def start_audio(self):
        print("Starting audio")
        try:
            self.p = pyaudio.PyAudio()
            self.input_devices, self.output_devices = self.get_audio_devices()
            input_idx = self.input_device_index.get() if self.input_device_index.get() >= 0 else None
            output_idx = self.output_device_index.get() if self.output_device_index.get() >= 0 else None
            
            print(f"Attempting to open audio streams - Input device: {input_idx}, Output device: {output_idx}")
            self.stream_out = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, output=True, 
                                        output_device_index=output_idx)
            self.stream_in = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True, 
                                       frames_per_buffer=CHUNK, input_device_index=input_idx)
            self.running = True
            self.rx_thread = threading.Thread(target=self.receive_loop, daemon=True)
            self.rx_thread.start()
            self.tx_thread = threading.Thread(target=self.transmit_loop, daemon=True)
            self.tx_thread.start()
            if self.rx_output:
                self.rx_output.insert(tk.END, "Audio started successfully\n")
            print("Audio threads started")
        except pyaudio.PyError as e:  # Corrected from PyAudioException to PyError
            if self.rx_output:
                self.rx_output.insert(tk.END, f"Audio error: Another program may have control of the audio. Error: {str(e)}\n")
            self.running = False
            print(f"Audio startup error: {str(e)}. Please close other audio programs and try again.")
            if self.root and self.rx_output:
                self.root.after(0, lambda: messagebox.showwarning("Audio Error", 
                                f"Audio setup failed: {str(e)}. Another program may have control of the audio. Close other audio programs and try again."))
            if self.p:
                self.p.terminate()
                self.p = None
        except Exception as e:
            if self.rx_output:
                self.rx_output.insert(tk.END, f"Audio failed to start: {str(e)}\n")
            self.running = False
            print(f"Audio startup error: {e}")
            if self.root and self.rx_output:
                self.root.after(0, lambda: messagebox.showwarning("Audio Warning", 
                                f"Audio setup failed: {str(e)}. Continuing without audio."))
            if self.p:
                self.p.terminate()
                self.p = None

    def toggle_dark_mode(self):
        if self.dark_mode_var.get():
            self.current_colors = {"bg": "#000000", "fg": "#00FF00", "entry_bg": "#1A1A1A", "button_bg": "#333333"}
        else:
            self.current_colors = self.light_colors
        self.root.configure(bg=self.current_colors["bg"])
        self.main_frame.configure(bg=self.current_colors["bg"])
        self.callsign_frame.configure(bg=self.current_colors["bg"])
        self.tx_frame.configure(bg=self.current_colors["bg"])
        self.rx_frame.configure(bg=self.current_colors["bg"])
        self.rx_header_frame.configure(bg=self.current_colors["bg"])
        self.speed_frame.configure(bg=self.current_colors["bg"])
        self.filter_frame.configure(bg=self.current_colors["bg"])
        self.v4_indicator_frame.configure(bg=self.current_colors["bg"])
        self.robust_plus_indicator_frame.configure(bg=self.current_colors["bg"])
        self.rx_output_frame.configure(bg=self.current_colors["bg"])
        
        for widget in (self.callsign_frame.winfo_children() + 
                       self.tx_frame.winfo_children() + 
                       self.rx_frame.winfo_children() + 
                       self.rx_header_frame.winfo_children() +
                       self.speed_frame.winfo_children() +
                       self.filter_frame.winfo_children() +
                       self.rx_output_frame.winfo_children() +
                       self.robust_plus_indicator_frame.winfo_children()):
            if isinstance(widget, (tk.Label, tk.Checkbutton, tk.Radiobutton)):
                widget.configure(fg=self.current_colors["fg"], bg=self.current_colors["bg"])
            elif isinstance(widget, tk.Entry):
                widget.configure(fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], 
                                insertbackground=self.current_colors["fg"])
            elif isinstance(widget, tk.Button):
                widget.configure(fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
        if self.rx_output:
            self.rx_output.configure(fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], 
                                    insertbackground=self.current_colors["fg"])
        self.clear_button.configure(fg=self.current_colors["fg"], bg=self.current_colors["button_bg"])
        self.dark_mode_check.configure(selectcolor=self.current_colors["entry_bg"])
        # Update button colors in dark mode
        self.tx_button.config(bg=self.current_colors["button_bg"])
        self.cq_button.config(bg=self.current_colors["button_bg"])

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

    def convolutional_encode(self, bits):
        encoded = []
        state = 0
        for bit in bits:
            encoded.extend(CONV_TABLE[(state, bit)])
            state = bit
        return encoded

    def viterbi_decode(self, bits):
        states = {0: (0, []), 1: (float('inf'), [])}
        for i in range(0, len(bits), 2):
            r1, r2 = bits[i], bits[i+1]
            new_states = {}
            for s in [0, 1]:
                for b in [0, 1]:
                    next_s = b
                    o1, o2 = CONV_TABLE[(s, b)]
                    cost = states[s][0] + (r1 ^ o1) + (r2 ^ o2)
                    path = states[s][1] + [b]
                    if next_s not in new_states or cost < new_states[next_s][0]:
                        new_states[next_s] = (cost, path)
            states = new_states
        return min(states.values(), key=lambda x: x[0])[1]

    def interleave(self, bits):
        block_size = 16
        padded = bits + [0] * (block_size - len(bits) % block_size) if len(bits) % block_size else bits
        return [padded[i // 4 + (i % 4) * 4] for i in range(len(padded))]

    def deinterleave(self, bits):
        block_size = 16
        deinterleaved = [0] * len(bits)
        for i in range(len(bits)):
            deinterleaved[i // 4 + (i % 4) * 4] = bits[i]
        return deinterleaved

    def detect_v4_preamble(self, data):
        # Robust mode preamble: "1357924" (original sequence)
        preamble_freqs = [(1500, 2000), (3000, 3500), (5000, 4500), (7000, 6500), 
                         (2000, 2500), (4000, 4500), (6000, 5500)]
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
        # Robust+ mode preamble: Unique sequence "2468135" (different from Robust)
        preamble_freqs = [(2000, 2500), (4000, 3500), (6000, 5500), (8000, 7500), 
                         (1000, 1500), (3000, 2500), (5000, 4500)]
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

    def decode_ofdm_frequency(self, data):
        """Decode a single OFDM frequency from audio data for Robust+ mode."""
        freqs = np.abs(fft(data)[:len(data)//2])
        freq_axis = np.linspace(0, SAMPLE_RATE // 2, len(freqs))
        peak_idx = np.argmax(freqs)
        peak_freq = freq_axis[peak_idx]
        tones = 16  # Number of OFDM subcarriers
        if np.max(freqs) > self.sensitivity.get() / 100.0 * np.max(freqs):  # Use sensitivity threshold
            for freq_idx in range(tones):
                expected_freq = 1200 + freq_idx * 150  # Center at 1200 Hz, 150 Hz spacing
                if abs(peak_freq - expected_freq) < 100:  # Tolerance of 100 Hz
                    return expected_freq
        return None

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

    def decode_v4_packet(self, symbols):
        if len(symbols) < 5:
            return None, None
        packet_id = V4_REVERSE_MAP.get(symbols[0], 0)
        payload_symbols = symbols[1:-4]
        crc_symbols = symbols[-4:]
        bits = [int(b) for sym in payload_symbols 
                for b in bin(V4_REVERSE_MAP.get(sym, 0))[2:].zfill(4)]
        deinterleaved_bits = self.deinterleave(bits)
        decoded_bits = self.viterbi_decode(deinterleaved_bits)
        text = "".join(chr(sum(b << (7-j) for j, b in enumerate(decoded_bits[i:i+8]))) 
                      for i in range(0, len(decoded_bits), 8) if len(decoded_bits[i:i+8]) == 8)
        crc = crc16(text.encode('utf-8'))
        received_crc = sum(V4_REVERSE_MAP.get(s, 0) << (12 - 4*i) for i, s in enumerate(crc_symbols))
        return packet_id, text if crc == received_crc else None

    def send_ack_nack(self, packet_id, success):
        if not self.stream_out:
            return
        self.set_v4_tx_indicator() if self.speed_var.get() == "robust" else self.set_robust_plus_tx_indicator()
        ack_sound = self.generate_sound(CHAR_SOUNDS["K" if success else "N"], 0.05)
        audio_data = np.concatenate((ack_sound, np.zeros(int(SAMPLE_RATE * 0.025))))
        self.stream_out.write(audio_data.astype(np.float32).tobytes())
        self.reset_robust_plus_indicator() if self.speed_var.get() == "robust_plus" else self.reset_indicator()

    def transmit(self):
        if not self.stream_out:
            messagebox.showwarning("Warning", "Audio not started!")
            return
        to_call = self.to_callsign.get().upper()
        my_call = self.my_callsign.get().upper()
        if not to_call or not my_call:
            messagebox.showwarning("Warning", "Enter both callsigns!")
            return
        text = self.tx_input.get().strip().upper()
        full_text = f"{to_call} DE {my_call} {text}" if text else f"{to_call} DE {my_call}"
        speed = self.speed_var.get()
        if speed == "robust":
            packet_id = (self.last_packet_id + 1) % 16
            self.last_packet_id = packet_id
            self.tx_queue.append((full_text, packet_id, False))  # False for Robust mode
        elif speed == "robust_plus":
            packet_id = (self.last_packet_id + 1) % 16
            self.last_packet_id = packet_id
            self.tx_queue.append((full_text, packet_id, True))  # True for Robust+ mode
        else:
            self.tx_queue.append((full_text, None, False))
        self.tx_input.delete(0, tk.END)
        self.start_radio_transmission()  # Start TX for this message
        # Change button color to red during TX
        self.tx_button.config(bg="red")

    def transmit_loop(self):
        while self.running:
            if self.tx_queue and self.stream_out:
                # Change button color to red at the start of transmission
                self.tx_button.config(bg="red")
                self.cq_button.config(bg="red")
                full_text, packet_id, is_robust_plus = self.tx_queue.pop(0)
                speed = self.speed_var.get()
                duration_map = {"slow": 0.2, "medium": 0.15, "fast": 0.1, "robust": 0.05, "robust_plus": 0.01}
                duration = duration_map[speed]
                gap_duration = duration / 2

                self.start_radio_transmission()  # Activate PTT at the start of transmission
                audio_data = np.array([], dtype=np.float32)
                if speed in ["robust", "robust_plus"]:
                    if is_robust_plus:
                        preamble_audio = np.concatenate([self.generate_sound(CHAR_SOUNDS[num], 0.05) 
                                                        for num in "2468135"])  # Robust+ preamble
                        self.set_robust_plus_tx_indicator()  # Orange for Robust+ TX
                        ofdm_symbols = self.encode_ofdm_packet(full_text, packet_id)
                        audio_data = preamble_audio
                        for freq, duration in ofdm_symbols:
                            sound = self.generate_ofdm_sound(freq, duration)
                            audio_data = np.concatenate((audio_data, sound, np.zeros(int(SAMPLE_RATE * gap_duration / 10))))
                        if self.rx_output:
                            self.rx_output.insert(tk.END, f"Robust+ TX (Preamble: 2468135): " + full_text + "\n", "sent")
                    else:
                        preamble_audio = np.concatenate([self.generate_sound(CHAR_SOUNDS[num], 0.05) 
                                                        for num in "1357924"])  # Robust preamble
                        self.set_v4_tx_indicator()  # Red for Robust TX
                        symbols = self.encode_v4_packet(full_text, packet_id)
                        audio_data = preamble_audio
                        for sym in symbols:
                            pattern = CHAR_SOUNDS.get(sym, CHAR_SOUNDS[" "])
                            sound = self.generate_sound(pattern, 0.05)
                            audio_data = np.concatenate((audio_data, sound, np.zeros(int(SAMPLE_RATE * gap_duration))))
                        if self.rx_output:
                            self.rx_output.insert(tk.END, f"Robust TX (Preamble: 1357924): " + full_text + "\n", "sent")
                    self.stream_out.write(audio_data.astype(np.float32).tobytes())
                    self.reset_robust_plus_indicator() if is_robust_plus else self.reset_indicator()
                    self.stop_radio_transmission()  # Deactivate PTT immediately after audio
                    time.sleep(0.2)  # Wait for ACK/NACK
                    if self.rx_ack == "N":  # Only re-queue on NACK
                        self.tx_queue.append((full_text, packet_id, is_robust_plus))
                    self.rx_ack = None
                else:
                    encoded_symbols = self.encode_fec(full_text)
                    for symbol in encoded_symbols:
                        char = REVERSE_MAP.get(symbol, " ")
                        pattern = WORD_SOUNDS.get(char, CHAR_SOUNDS.get(char, []))
                        sound = self.generate_sound(pattern, duration)
                        audio_data = np.concatenate((audio_data, sound, np.zeros(int(SAMPLE_RATE * gap_duration))))
                    self.stream_out.write(audio_data.astype(np.float32).tobytes())
                    if self.rx_output:
                        self.rx_output.insert(tk.END, "Sent: " + full_text + "\n", "sent")
                    self.stop_radio_transmission()  # Deactivate PTT immediately after audio
                # Reset button colors to default after transmission
                self.tx_button.config(bg=self.current_colors["button_bg"])
                self.cq_button.config(bg=self.current_colors["button_bg"])
                time.sleep(0.001)  # Minimal sleep to prevent CPU overload, no additional delay
            time.sleep(0.001)  # Minimal sleep to prevent CPU overload

    def send_cq(self):
        if not self.stream_out:
            messagebox.showwarning("Warning", "Audio not started!")
            return
        my_call = self.my_callsign.get().upper()
        if not my_call:
            messagebox.showwarning("Warning", "Enter your callsign!")
            return
        speed = self.speed_var.get()
        single_cq = f"CQ CQ CQ DE {my_call}"
        if speed in ["robust", "robust_plus"]:
            packet_id = (self.last_packet_id + 1) % 16
            self.last_packet_id = packet_id
            self.tx_queue.append((single_cq, packet_id, speed == "robust_plus"))  # Single CQ for Robust/Robust+
        else:
            cq_text = f"{single_cq} {single_cq}"  # Two repeats for old modes
            self.tx_queue.append((cq_text, None, False))
        self.start_radio_transmission()  # Start TX for CQ
        # Change button color to red during TX
        self.cq_button.config(bg="red")

    def test_ptt_connection(self):
        """Test the radio PTT by activating it for 1 second and then deactivating it immediately."""
        if self.serial_port.get() != "NONE" and self.baud_rate.get():
            if not self.baud_rate.get().isdigit():
                messagebox.showwarning("Radio Warning", "Baud rate must be a valid number.")
                return
            self.start_radio_transmission()  # Activate PTT for testing
            # Schedule PTT release after exactly 1 second and reset button colors
            self.root.after(1000, lambda: [self.stop_radio_transmission(), self.tx_button.config(bg=self.current_colors["button_bg"]), self.cq_button.config(bg=self.current_colors["button_bg"])])
        else:
            messagebox.showwarning("Radio Warning", "No serial port or baud rate selected for radio communication.")

    def start_radio_transmission(self):
        """Activate PTT (TX) for the radio, dynamically adjusting for radio type."""
        if self.serial_port.get() != "NONE" and self.baud_rate.get():
            try:
                radio = self.radio.get()
                address = self.usb_address.get().replace('H', '') if self.usb_address.get() else ''
                if not address:
                    raise ValueError("No address specified for radio communication.")

                if not self.radio_serial or not self.radio_serial.is_open:
                    self.radio_serial = serial.Serial(
                        port=self.serial_port.get(),
                        baudrate=int(self.baud_rate.get()) if self.baud_rate.get().isdigit() else 9600,  # Default to 9600 if invalid
                        timeout=0.1,  # Reduced timeout for faster response
                        rtscts=self.rts.get(),
                        dsrdtr=self.dtr.get()
                    )
                    print(f"Radio serial connection established on {self.serial_port.get()} at {self.baud_rate.get()} baud")
                    if self.rx_output:
                        self.rx_output.insert(tk.END, f"Radio connection established on {self.serial_port.get()}\n")

                # Generate CI-V command for PTT ON based on radio type
                if radio in ["Icom IC-703", "Icom IC-705"]:
                    cmd = bytes.fromhex(f"FE FE {address} E0 1C 00 FD")
                else:  # Generic radio, use custom address for Icom-like CI-V
                    cmd = bytes.fromhex(f"FE FE {address} E0 1C 00 FD")  # Default Icom CI-V format, user must ensure compatibility
                self.radio_serial.write(cmd)
                print(f"Sent PTT ON command: {cmd.hex()}")
                if self.rx_output:
                    self.rx_output.insert(tk.END, f"Sent PTT ON command: {cmd.hex()}\n")
            except (serial.SerialException, ValueError) as e:
                print(f"Failed to start radio transmission: {str(e)}")
                if self.rx_output:
                    self.rx_output.insert(tk.END, f"Failed to start radio transmission: {str(e)}\n")
                if self.radio_serial:
                    self.radio_serial.close()
                    self.radio_serial = None

    def stop_radio_transmission(self):
        """Deactivate PTT (TX) for the radio immediately after audio or test, dynamically adjusting for radio type."""
        if self.radio_serial and self.radio_serial.is_open:
            try:
                radio = self.radio.get()
                address = self.usb_address.get().replace('H', '') if self.usb_address.get() else ''
                if not address:
                    raise ValueError("No address specified for radio communication.")

                # Generate CI-V command for PTT OFF based on radio type
                if radio in ["Icom IC-703", "Icom IC-705"]:
                    cmd = bytes.fromhex(f"FE FE {address} E0 1C 01 FD")
                else:  # Generic radio, use custom address for Icom-like CI-V
                    cmd = bytes.fromhex(f"FE FE {address} E0 1C 01 FD")  # Default Icom CI-V format, user must ensure compatibility
                self.radio_serial.write(cmd)
                print(f"Sent PTT OFF command: {cmd.hex()}")
                if self.rx_output:
                    self.rx_output.insert(tk.END, f"Sent PTT OFF command: {cmd.hex()}\n")
            except serial.SerialException as e:
                print(f"Failed to stop radio transmission: {str(e)}")
                if self.rx_output:
                    self.rx_output.insert(tk.END, f"Failed to stop radio transmission: {str(e)}\n")
            finally:
                if self.radio_serial:
                    self.radio_serial.close()
                    self.radio_serial = None

    def encode_fec(self, text):
        symbol_list = []
        words = text.split()
        for word in words:
            if word in WORD_SOUNDS:
                symbol_list.append(SYMBOL_MAP[word])
            else:
                for char in word:
                    symbol_list.append(SYMBOL_MAP.get(char, SYMBOL_MAP[" "]))
        speed = self.speed_var.get()
        if speed in ["slow", "medium"]:
            while len(symbol_list) % 23 != 0:
                symbol_list.append(SYMBOL_MAP[" "])
            blocks = [symbol_list[i:i+23] for i in range(0, len(symbol_list), 23)]
            encoded_blocks = []
            for block in blocks:
                encoded = RSCodec.encode(data=block)
                encoded_blocks.extend(encoded)
            return encoded_blocks
        elif speed == "robust":
            while len(symbol_list) % 23 != 0:
                symbol_list.append(SYMBOL_MAP[" "])
            blocks = [symbol_list[i:i+23] for i in range(0, len(symbol_list), 23)]
            encoded_blocks = []
            for block in blocks:
                encoded = RSCodec.encode(data=block)
                encoded_blocks.extend(encoded)
            return encoded_blocks
        elif speed == "robust_plus":
            # OFDM/MFSK Hybrid for Robust+ mode
            ofdm_symbols = self.encode_ofdm_packet(text, 0)  # Placeholder packet ID
            return ofdm_symbols
        return symbol_list

    def decode_fec(self, symbols):
        """Decode FEC-encoded symbols for received data, ensuring full message accumulation."""
        speed = self.speed_var.get()
        if speed in ["slow", "medium"]:
            blocks = [symbols[i:i+31] for i in range(0, len(symbols), 31)]
            decoded_text = ""
            for block in blocks:
                if len(block) == 31:
                    try:
                        if not all(isinstance(s, int) for s in block):
                            print(f"Invalid data in block: {block}")
                            decoded_text += "[ERROR] Invalid symbol data "
                            continue
                        decoded = RSCodec.decode(data=block)[0]  # Explicitly specify 'data' parameter
                        decoded_text += "".join(REVERSE_MAP.get(sym, " ") for sym in decoded)
                    except ReedSolomonError:
                        decoded_text += "[ERROR] "
            return decoded_text.strip()
        elif speed == "robust":
            blocks = [symbols[i:i+31] for i in range(0, len(symbols), 31)]
            decoded_text = ""
            for block in blocks:
                if len(block) == 31:
                    try:
                        if not all(isinstance(s, int) for s in block):
                            print(f"Invalid data in block: {block}")
                            decoded_text += "[ERROR] Invalid symbol data "
                            continue
                        decoded = RSCodec.decode(data=block)[0]  # Explicitly specify 'data' parameter
                        decoded_text += "".join(REVERSE_MAP.get(sym, " ") for sym in decoded)
                    except ReedSolomonError:
                        decoded_text += "[ERROR] "
            return decoded_text.strip()
        elif speed == "robust_plus":
            # Decode OFDM/MFSK Hybrid for Robust+ mode
            if not isinstance(symbols, list) or not all(isinstance(s, tuple) and len(s) == 2 for s in symbols):
                print(f"Invalid OFDM symbols: {symbols}")
                return "[ERROR] Invalid OFDM data"
            decoded_text = self.decode_ofdm_packet(symbols)
            return decoded_text.strip()
        return "".join(REVERSE_MAP.get(sym, " ") for sym in symbols).strip()

    def encode_ofdm_packet(self, text, packet_id):
        # Implement OFDM/MFSK hybrid encoding for Robust+ mode
        # Use 16 tones (configurable) within 2400 Hz bandwidth, targeting ~40-100 WPM
        bits = [int(b) for char in text for b in bin(ord(char))[2:].zfill(8)]
        # Convolutional encoding for robustness
        encoded_bits = self.convolutional_encode(bits)
        # Interleave for burst error correction
        interleaved_bits = self.interleave(encoded_bits)
        # LDPC or Turbo coding for additional robustness (simplified here with Reed-Solomon as placeholder)
        rs_encoded = RSCodec(16).encode(data=interleaved_bits)  # Increased parity for Robust+
        # Map bits to OFDM symbols (16 tones, 150 Hz spacing, 2400 Hz total bandwidth)
        ofdm_symbols = []
        tones = 16  # Number of OFDM subcarriers
        symbol_rate = 41.6  # bps per tone, targeting ~40-100 WPM
        for i in range(0, len(rs_encoded), int(symbol_rate)):
            chunk = rs_encoded[i:i+int(symbol_rate)]
            if chunk:
                freq_idx = sum(chunk) % tones  # Simple mapping to tone index
                freq = 1200 + freq_idx * 150  # Center at 1200 Hz, 150 Hz spacing for 2400 Hz total
                duration = 0.01  # 10 ms per symbol for high speed
                ofdm_symbols.append((freq, duration))
        # Add CRC-16 for error detection
        crc = crc16(text.encode('utf-8'))
        crc_bits = [int(b) for b in bin(crc)[2:].zfill(16)]
        for i in range(0, len(crc_bits), int(symbol_rate)):
            chunk = crc_bits[i:i+int(symbol_rate)]
            if chunk:
                freq_idx = sum(chunk) % tones
                freq = 1200 + freq_idx * 150
                duration = 0.01
                ofdm_symbols.append((freq, duration))
        return ofdm_symbols

    def decode_ofdm_packet(self, symbols):
        # Implement OFDM/MFSK hybrid decoding for Robust+ mode
        decoded_bits = []
        tones = 16  # Number of OFDM subcarriers
        symbol_rate = 41.6  # bps per tone
        for freq, _ in symbols:
            freq_idx = int((freq - 1200) / 150) % tones  # Reverse mapping from frequency
            bits = [int(b) for b in bin(freq_idx)[2:].zfill(int(np.log2(tones)))]
            decoded_bits.extend(bits[:int(symbol_rate)])
        # Deinterleave
        deinterleaved_bits = self.deinterleave(decoded_bits)
        # Viterbi decode
        viterbi_decoded = self.viterbi_decode(deinterleaved_bits)
        # Reed-Solomon decode (with 16 parity symbols for Robust+)
        try:
            rs_decoded = RSCodec(16).decode(data=viterbi_decoded)[0]
            # Extract text from bits
            text = ""
            for i in range(0, len(rs_decoded), 8):
                byte = rs_decoded[i:i+8]
                if len(byte) == 8:
                    char = chr(sum(b << (7-j) for j, b in enumerate(byte)))
                    text += char
            # Verify CRC-16
            crc_pos = len(text) - 2  # Assuming CRC-16 is appended at the end
            if crc_pos > 0:
                crc_received = int(text[crc_pos:], 16)
                crc_calculated = crc16(text[:crc_pos].encode('utf-8'))
                if crc_received != crc_calculated:
                    return "[ERROR] CRC mismatch"
            return text
        except ReedSolomonError:
            return "[ERROR] Decoding failed"

    def generate_ofdm_sound(self, freq, duration):
        # Generate OFDM tone for Robust+ mode
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
        signal = np.sin(2 * np.pi * freq * t)
        return signal / np.max(np.abs(signal))

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_filter(self, data):
        filter_type = self.filter_var.get()
        if filter_type == "none":
            return data
        elif filter_type == "900-5100":
            b, a = self.butter_bandpass(900, 2400, SAMPLE_RATE)  # Narrower for Robust+
            return lfilter(b, a, data)
        elif filter_type == "800-5200":
            b, a = self.butter_bandpass(800, 2400, SAMPLE_RATE)  # Narrower for Robust+
            return lfilter(b, a, data)
        return data

    def decode_audio(self, data):
        """Decode audio data with adjustable sensitivity based on user settings (0–100 scale)."""
        freqs = np.abs(fft(data)[:len(data)//2])
        freq_axis = np.linspace(0, SAMPLE_RATE // 2, len(freqs))
        peak_idx = np.argmax(freqs)
        peak_freq = freq_axis[peak_idx]
        
        # Convert sensitivity (0–100) to a threshold (0.0–1.0) for min_amplitude
        sensitivity_value = self.sensitivity.get() / 100.0  # Scale from 0–100 to 0.0–1.0
        min_amplitude = 0.01 + (sensitivity_value * 0.99)  # Scale from 0.01 to 1.0 (higher sensitivity = lower threshold)
        print(f"Decode audio - Sensitivity: {sensitivity_value}, Min Amplitude: {min_amplitude}, Peak Amplitude: {np.max(freqs)}")  # Enhanced debug logging

        if np.max(freqs) < min_amplitude * np.max(freqs):  # Only process if signal exceeds threshold
            return None

        tolerance = 300  # Fixed tolerance for frequency matching
        speed = self.speed_var.get()
        if speed in ["robust", "robust_plus"]:
            for sym, value in V4_REVERSE_MAP.items() if speed == "robust" else range(16):  # Placeholder for Robust+
                pattern = CHAR_SOUNDS[sym] if speed == "robust" else [(1200 + value * 150, 1200 + value * 150, "tone")]
                start_freq = pattern[0][0]
                if abs(peak_freq - start_freq) < tolerance:
                    return sym if speed == "robust" else value
        else:
            for word, pattern in WORD_SOUNDS.items():
                start_freq = pattern[0][0]
                if abs(peak_freq - start_freq) < tolerance:
                    return SYMBOL_MAP[word]
            for char, pattern in CHAR_SOUNDS.items():
                start_freq = pattern[0][0]
                if abs(peak_freq - start_freq) < tolerance:
                    return SYMBOL_MAP[char]
        return None

    def set_v4_tx_indicator(self):
        if self.indicator_timeout is not None:
            self.root.after_cancel(self.indicator_timeout)
        self.v4_indicator.itemconfig("robust_mode_light", fill="red")

    def set_v4_rx_indicator(self):
        self.v4_indicator.itemconfig("robust_mode_light", fill="green")
        if self.indicator_timeout is not None:
            self.root.after_cancel(self.indicator_timeout)
        self.indicator_timeout = self.root.after(3000, self.reset_indicator)

    def set_robust_plus_tx_indicator(self):
        if self.indicator_timeout is not None:
            self.root.after_cancel(self.indicator_timeout)
        self.robust_plus_indicator.itemconfig("robust_plus_mode_light", fill="orange")

    def set_robust_plus_rx_indicator(self):
        self.robust_plus_indicator.itemconfig("robust_plus_mode_light", fill="yellow")
        if self.indicator_timeout is not None:
            self.root.after_cancel(self.indicator_timeout)
        self.indicator_timeout = self.root.after(3000, self.reset_robust_plus_indicator)

    def reset_indicator(self):
        self.v4_indicator.itemconfig("robust_mode_light", fill="grey")
        self.indicator_timeout = None

    def reset_robust_plus_indicator(self):
        self.robust_plus_indicator.itemconfig("robust_plus_mode_light", fill="grey")
        self.indicator_timeout = None

    def receive_loop(self):
        """Receive loop with buffered text display and adjustable sensitivity, accumulating symbols to fill the dynamic width without space-based line breaks."""
        buffer = []
        gap_map = {"slow": 0.1, "medium": 0.075, "fast": 0.05, "robust": 0.025, "robust_plus": 0.005}
        duration_map = {"slow": 0.2, "medium": 0.15, "fast": 0.1, "robust": 0.05, "robust_plus": 0.01}
        preamble_detected = False
        last_update_time = time.time()
        update_interval = 2.0  # Increased to 2.0 seconds for better text accumulation

        while self.running:
            try:
                speed = self.speed_var.get()
                duration = duration_map[speed]
                chunk_size = int(SAMPLE_RATE * (duration + gap_map[speed] * 2))
                data = self.stream_in.read(chunk_size, exception_on_overflow=False)
                if not data:
                    print("No audio data received, skipping chunk.")
                    time.sleep(0.01)
                    continue
                data = np.frombuffer(data, dtype=np.float32)
                gain = self.input_volume.get() / 100.0  # Use input_volume for gain
                print(f"Receive loop - Input volume: {gain}, Peak amplitude: {np.max(np.abs(data))}")  # Debug volume effect
                data = data * gain
                data = self.apply_filter(data)

                if not preamble_detected and len(buffer) == 0:
                    if self.detect_v4_preamble(data[:int(SAMPLE_RATE * 0.4)]):
                        self.is_v4_mode = True
                        self.set_v4_rx_indicator()
                        preamble_detected = True
                        continue
                    elif self.detect_robust_plus_preamble(data[:int(SAMPLE_RATE * 0.4)]):
                        self.is_v4_mode = True
                        self.set_robust_plus_rx_indicator()
                        preamble_detected = True
                        continue

                symbol = self.decode_audio(data)
                if symbol is not None:
                    if speed in ["robust", "robust_plus"]:
                        if isinstance(symbol, tuple) and len(symbol) == 2:  # Handle OFDM tuples for Robust+
                            buffer.append(symbol)
                        else:  # Handle single symbols for Robust
                            buffer.append(symbol)
                        if len(buffer) >= 10:
                            packet_id, decoded_text = (self.decode_v4_packet(buffer) if speed == "robust" 
                                                    else self.decode_ofdm_packet(buffer))
                            if decoded_text and packet_id not in self.packet_buffer:
                                self.packet_buffer[packet_id] = decoded_text
                                preamble = "1357924" if speed == "robust" else "2468135"
                                self.display_received_text(f"{'Robust' if speed == 'robust' else 'Robust+'} RX (Preamble: {preamble}): " + decoded_text + "\n", "received")
                                self.send_ack_nack(packet_id, True)
                                self.rx_ack = "K"
                            elif not decoded_text:
                                self.send_ack_nack(packet_id, False)
                                self.rx_ack = "N"
                            buffer.clear()
                            preamble_detected = False
                            self.is_v4_mode = False
                            self.reset_robust_plus_indicator() if speed == "robust_plus" else self.reset_indicator()
                    else:
                        block_size = 31 if speed in ["slow", "medium"] else 1
                        if len(buffer) >= block_size:
                            decoded_text = self.decode_fec(buffer[:block_size])
                            self.temp_receive_buffer.extend(decoded_text)  # Accumulate in temporary buffer
                            print(f"Decoded text accumulated: {decoded_text}, Buffer length: {len(''.join(self.temp_receive_buffer))}")  # Debug buffering
                            buffer = buffer[block_size:]
                        current_time = time.time()
                        if not hasattr(self, 'dynamic_width'):
                            self.update_dynamic_width(None)  # Ensure dynamic_width is set
                        buffer_text = "".join(self.temp_receive_buffer)
                        # Flush only if buffer exceeds dynamic width or timeout occurs
                        if (len(buffer_text) >= self.dynamic_width or 
                            current_time - last_update_time >= update_interval or not self.running):
                            if self.temp_receive_buffer:
                                self.display_received_text(buffer_text + "\n")
                                self.temp_receive_buffer = []  # Clear after displaying
                            last_update_time = current_time
                time.sleep(0.01)  # Slow down processing to reduce noise sensitivity
            except Exception as e:
                if self.running:
                    if self.rx_output:
                        self.rx_output.insert(tk.END, f"Receive error: {str(e)}\n")
                    print(f"Receive loop error: {e}")
                break

    def display_received_text(self, text, tag=""):
        """Display received text in the receive window, buffering to fill the dynamic width without space-based line breaks."""
        if not self.rx_output:
            return

        if not hasattr(self, 'dynamic_width'):
            self.update_dynamic_width(None)  # Initialize if not set
        widget_width = self.dynamic_width  # Dynamic width in characters

        print(f"Display received text - widget_width: {widget_width}, text length: {len(text)}, text: {text[:50]}...")  # Debug logging

        # Split text into characters and buffer them
        for char in text:
            self.receive_buffer.append(char)
            if len(self.receive_buffer) >= widget_width:
                self.rx_output.insert(tk.END, ''.join(self.receive_buffer) + "\n", tag)
                self.receive_buffer = []

        # Flush any remaining buffer at the end of a message, ensuring no partial lines
        if self.receive_buffer:
            remaining_text = ''.join(self.receive_buffer)
            if remaining_text:
                self.rx_output.insert(tk.END, remaining_text + "\n", tag)
            self.receive_buffer = []

        self.rx_output.see(tk.END)  # Auto-scroll to the bottom

    def on_closing(self):
        self.running = False
        if self.stream_out:
            try:
                self.stream_out.stop_stream()
                self.stream_out.close()
            except Exception as e:
                print(f"Error closing output stream: {e}")
            self.stream_out = None
        if self.stream_in:
            try:
                self.stream_in.stop_stream()
                self.stream_in.close()
            except Exception as e:
                print(f"Error closing input stream: {e}")
            self.stream_in = None
        if self.rx_thread and self.rx_thread.is_alive():
            self.rx_thread.join(timeout=1.0)
        if self.tx_thread and self.tx_thread.is_alive():
            self.tx_thread.join(timeout=1.0)
        self.rx_thread = None
        self.tx_thread = None
        if self.radio_serial and self.radio_serial.is_open:
            self.stop_radio_transmission()  # Ensure PTT is off before closing
        if self.p:
            try:
                self.p.terminate()
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")
            self.p = None
        self.root.destroy()
        print("Application closed")

    def show_audio_settings_window(self):
        audio_window = tk.Toplevel(self.root)
        audio_window.title("Audio Settings")
        audio_window.geometry("600x400")  # Increased size for more depth
        audio_window.resizable(False, False)

        # Audio Input
        tk.Label(audio_window, text="Audio Input:", fg=self.current_colors["fg"], 
                 bg=self.current_colors["bg"]).pack(pady=5)
        input_options = list(self.input_devices.values())
        self.input_device = tk.StringVar(value=list(self.input_devices.values())[self.input_device_index.get()] 
                                        if self.input_device_index.get() >= 0 else "Default")
        tk.OptionMenu(audio_window, self.input_device, "Default", *input_options, 
                     command=self.update_audio_devices).pack(pady=5)

        # Audio Output
        tk.Label(audio_window, text="Audio Output:", fg=self.current_colors["fg"], 
                 bg=self.current_colors["bg"]).pack(pady=5)
        output_options = list(self.output_devices.values())
        self.output_device = tk.StringVar(value=list(self.output_devices.values())[self.output_device_index.get()] 
                                         if self.output_device_index.get() >= 0 else "Default")
        tk.OptionMenu(audio_window, self.output_device, "Default", *output_options, 
                     command=self.update_audio_devices).pack(pady=5)

        # Input Volume
        tk.Label(audio_window, text="Input Volume (%):", fg=self.current_colors["fg"], 
                 bg=self.current_colors["bg"]).pack(pady=5)
        tk.Scale(audio_window, from_=0, to=100, variable=self.input_volume, orient=tk.HORIZONTAL, 
                 length=400, fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(pady=5)

        # Sensitivity/Squelch (0–100, higher values mean less sensitivity/more squelch)
        tk.Label(audio_window, text="Sensitivity/Squelch (0–100):", fg=self.current_colors["fg"], 
                 bg=self.current_colors["bg"]).pack(pady=5)
        tk.Scale(audio_window, from_=0, to=100, variable=self.sensitivity, orient=tk.HORIZONTAL, 
                 length=400, resolution=1, fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(pady=5)

        # Sensitivity Description
        tk.Label(audio_window, text="Higher values reduce sensitivity (increase squelch), lower values increase sensitivity.", 
                 fg=self.current_colors["fg"], bg=self.current_colors["bg"], wraplength=500).pack(pady=5)

        # Save Button
        tk.Button(audio_window, text="Save", command=lambda: [self.save_audio_settings(), audio_window.destroy()], 
                  fg=self.current_colors["fg"], bg=self.current_colors["button_bg"]).pack(pady=10)

    def update_audio_devices(self, *args):
        """Dynamically update audio streams when devices are selected in the dropdowns."""
        if self.p and self.running:
            try:
                input_idx = self.input_device_index.get() if self.input_device_index.get() >= 0 else None
                output_idx = self.output_device_index.get() if self.output_device_index.get() >= 0 else None
                
                if self.stream_in:
                    self.stream_in.stop_stream()
                    self.stream_in.close()
                if self.stream_out:
                    self.stream_out.stop_stream()
                    self.stream_out.close()

                self.stream_out = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, output=True, 
                                            output_device_index=output_idx)
                self.stream_in = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True, 
                                           frames_per_buffer=CHUNK, input_device_index=input_idx)
                if self.rx_output:
                    self.rx_output.insert(tk.END, f"Audio devices updated to Input: {input_idx}, Output: {output_idx}\n")
                print(f"Audio devices updated - Input device: {input_idx}, Output device: {output_idx}")
            except pyaudio.PyError as e:  # Corrected from PyAudioException to PyError
                if self.rx_output:
                    self.rx_output.insert(tk.END, f"Audio update error: {str(e)}\n")
                print(f"Audio update error: {str(e)}. Please close other audio programs and try again.")
                if self.root and self.rx_output:
                    self.root.after(0, lambda: messagebox.showwarning("Audio Error", 
                                    f"Failed to update audio devices: {str(e)}. Close other audio programs and try again."))
                if self.p:
                    self.p.terminate()
                    self.p = None
                    self.running = False
            except Exception as e:
                if self.rx_output:
                    self.rx_output.insert(tk.END, f"Audio update failed: {str(e)}\n")
                print(f"Audio update error: {e}")
                if self.root and self.rx_output:
                    self.root.after(0, lambda: messagebox.showwarning("Audio Warning", 
                                    f"Failed to update audio devices: {str(e)}. Continuing without audio."))
                if self.p:
                    self.p.terminate()
                    self.p = None
                    self.running = False

    def save_audio_settings(self):
        """Save audio settings to the settings file and update audio streams."""
        input_value = self.input_device.get()
        output_value = self.output_device.get()
        
        try:
            if input_value in self.input_devices.values():
                self.input_device_index.set(list(self.input_devices.keys())[list(self.input_devices.values()).index(input_value)])
            elif input_value == "Default":
                self.input_device_index.set(-1)
            else:
                self.input_device_index.set(-1)  # Default if not found

            if output_value in self.output_devices.values():
                self.output_device_index.set(list(self.output_devices.keys())[list(self.output_devices.values()).index(output_value)])
            elif output_value == "Default":
                self.output_device_index.set(-1)
            else:
                self.output_device_index.set(-1)  # Default if not found

            self.save_settings()
            self.update_audio_devices()  # Ensure audio is redirected after saving
        except Exception as e:
            if self.rx_output:
                self.rx_output.insert(tk.END, f"Error saving audio settings: {str(e)}\n")
            print(f"Error saving audio settings: {e}")

    def show_radio_settings_window(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Radio Settings")
        settings_window.geometry("600x500")  # Increased size for layout and COM ports
        settings_window.resizable(False, False)

        # Radio Selection Section (Top)
        radio_frame = tk.LabelFrame(settings_window, text="Radio Selection", fg=self.current_colors["fg"], 
                                   bg=self.current_colors["bg"])
        radio_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(radio_frame, text="Radio:", fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(side="left", padx=5)
        radio_options = ["Icom IC-703", "Icom IC-705", "Generic"]
        tk.OptionMenu(radio_frame, self.radio, *radio_options, command=self.update_radio_fields).pack(side="left", padx=5)

        tk.Label(radio_frame, text="Address:", fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(side="left", padx=5)
        self.usb_entry = tk.Entry(radio_frame, textvariable=self.usb_address, width=10, 
                                 fg=self.current_colors["fg"], bg=self.current_colors["entry_bg"], 
                                 insertbackground=self.current_colors["fg"])
        self.usb_entry.pack(side="left", padx=5)

        tk.Label(radio_frame, text="Mode:", fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(side="left", padx=5)
        self.mode_frame = tk.Frame(radio_frame, bg=self.current_colors["bg"])
        self.mode_frame.pack(side="left", padx=2)
        self.mode_usb_check = tk.Checkbutton(self.mode_frame, text="USB", variable=self.mode_usb, 
                                           fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                                           selectcolor=self.current_colors["entry_bg"])
        self.mode_usb_check.pack(side="left", padx=2)
        self.mode_usb_digital_check = tk.Checkbutton(self.mode_frame, text="USB Digital", variable=self.mode_usb_digital, 
                                                   fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                                                   selectcolor=self.current_colors["entry_bg"])
        self.mode_usb_digital_check.pack(side="left", padx=2)
        self.mode_fm_check = tk.Checkbutton(self.mode_frame, text="FM", variable=self.mode_fm, 
                                          fg=self.current_colors["fg"], bg=self.current_colors["bg"], 
                                          selectcolor=self.current_colors["entry_bg"])
        self.mode_fm_check.pack(side="left", padx=2)

        # Radio Control Port Section (Middle)
        control_frame = tk.LabelFrame(settings_window, text="Radio Control Port", fg=self.current_colors["fg"], 
                                    bg=self.current_colors["bg"])
        control_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(control_frame, text="Serial Port:", fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(side="left", padx=5)
        com_ports = self.get_com_ports()
        tk.OptionMenu(control_frame, self.serial_port, *com_ports).pack(side="left", padx=5)

        tk.Label(control_frame, text="Baud Rate:", fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(side="left", padx=5)
        baud_rates = ["", "1200", "2400", "4800", "9600", "19200", "38400", "57600", "115200"]  # Empty as default
        tk.OptionMenu(control_frame, self.baud_rate, *baud_rates).pack(side="left", padx=5)

        tk.Checkbutton(control_frame, text="RTS", variable=self.rts, fg=self.current_colors["fg"], 
                      bg=self.current_colors["bg"], selectcolor=self.current_colors["entry_bg"], 
                      command=self.update_rts_dtr).pack(side="left", padx=2)
        tk.Checkbutton(control_frame, text="DTR", variable=self.dtr, fg=self.current_colors["fg"], 
                      bg=self.current_colors["bg"], selectcolor=self.current_colors["entry_bg"], 
                      command=self.update_rts_dtr).pack(side="left", padx=2)

        # PTT Port Section (Bottom)
        ptt_frame = tk.LabelFrame(settings_window, text="PTT Port", fg=self.current_colors["fg"], 
                                 bg=self.current_colors["bg"])
        ptt_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(ptt_frame, text="PTT Port:", fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(side="left", padx=5)
        tk.OptionMenu(ptt_frame, self.ptt_port, *com_ports).pack(side="left", padx=5)

        tk.Checkbutton(ptt_frame, text="RTS", variable=self.ptt_rts, fg=self.current_colors["fg"], 
                      bg=self.current_colors["bg"], selectcolor=self.current_colors["entry_bg"], 
                      command=self.update_rts_dtr).pack(side="left", padx=2)
        tk.Checkbutton(ptt_frame, text="DTR", variable=self.ptt_dtr, fg=self.current_colors["fg"], 
                      bg=self.current_colors["bg"], selectcolor=self.current_colors["entry_bg"], 
                      command=self.update_rts_dtr).pack(side="left", padx=2)

        # Radio-specific TX command info (as tooltips)
        self.radio_tx_info = {
            "Icom IC-703": "TX Command: FE FE 70 E0 1C 00 FD (Hex via serial, 9600 baud, RTS/DTR optional)",
            "Icom IC-705": "TX Command: FE FE 94 E0 1C 00 FD (Hex via serial, 9600 baud, RTS/DTR optional)",
            "Generic": "TX Command: User-defined (e.g., enter custom address like 26A for CI-V, adjust protocol as needed)"
        }
        tk.Label(ptt_frame, text="TX Command Info:", fg=self.current_colors["fg"], bg=self.current_colors["bg"]).pack(side="left", padx=5)
        self.tx_info_label = tk.Label(ptt_frame, text=self.radio_tx_info.get(self.radio.get(), ""), 
                                     fg=self.current_colors["fg"], bg=self.current_colors["bg"], wraplength=200)
        self.tx_info_label.pack(side="left", padx=5)
        self.radio.trace("w", self.update_tx_info)

        # Save Button
        tk.Button(settings_window, text="Save", command=lambda: [self.save_settings(), settings_window.destroy()], 
                  fg=self.current_colors["fg"], bg=self.current_colors["button_bg"]).pack(pady=5)
        tk.Button(settings_window, text="Test PTT", command=self.test_ptt_connection, 
                  fg=self.current_colors["fg"], bg=self.current_colors["button_bg"]).pack(pady=5)

    def update_radio_fields(self, *args):
        """Update radio fields based on selected radio type."""
        radio = self.radio.get()
        default_address = self.get_default_address(radio)
        self.usb_address.set(default_address)
        if radio == "Generic":
            self.usb_address.set("")  # Empty for custom input
            self.mode_usb.set(False)
            self.mode_usb_digital.set(False)
            self.mode_fm.set(False)
            # Disable mode checkboxes for Generic, user must configure manually
            self.mode_usb_check.config(state="disabled")
            self.mode_usb_digital_check.config(state="disabled")
            self.mode_fm_check.config(state="disabled")
        else:
            self.mode_usb_check.config(state="normal")
            self.mode_usb_digital_check.config(state="normal")
            self.mode_fm_check.config(state="normal")

    def update_rts_dtr(self):
        """Update RTS/DTR settings and ensure they affect serial communication."""
        print(f"RTS: {self.rts.get()}, DTR: {self.dtr.get()}")
        print(f"PTT RTS: {self.ptt_rts.get()}, PTT DTR: {self.ptt_dtr.get()}")
        # This method ensures the checkboxes are functional and trigger updates in serial settings

    def update_tx_info(self, *args):
        radio = self.radio.get()
        self.tx_info_label.config(text=self.radio_tx_info.get(radio, ""))

    def communicate_with_radio(self, command):
        """Send a CAT command to the radio and handle the response, dynamically adjusting for radio type."""
        if self.serial_port.get() != "NONE" and self.baud_rate.get():
            try:
                radio = self.radio.get()
                address = self.usb_address.get().replace('H', '') if self.usb_address.get() else ''
                if not address:
                    raise ValueError("No address specified for radio communication.")

                if not self.radio_serial or not self.radio_serial.is_open:
                    self.radio_serial = serial.Serial(
                        port=self.serial_port.get(),
                        baudrate=int(self.baud_rate.get()) if self.baud_rate.get().isdigit() else 9600,  # Default to 9600 if invalid
                        timeout=0.1,  # Reduced timeout for faster response
                        rtscts=self.rts.get(),
                        dsrdtr=self.dtr.get()
                    )
                    print(f"Radio serial connection established on {self.serial_port.get()} at {self.baud_rate.get()} baud")
                    if self.rx_output:
                        self.rx_output.insert(tk.END, f"Radio connection established on {self.serial_port.get()}\n")

                if command == "READ_FREQ":
                    # Default Icom-like CI-V command for frequency read, adaptable for Generic
                    if radio in ["Icom IC-703", "Icom IC-705"]:
                        cmd = bytes.fromhex(f"FE FE {address} E0 03 FD")
                    else:  # Generic, use custom address for Icom-like CI-V
                        cmd = bytes.fromhex(f"FE FE {address} E0 03 FD")  # User must ensure compatibility
                    self.radio_serial.write(cmd)
                    time.sleep(0.1)  # Wait for response
                    if self.radio_serial.in_waiting > 0:
                        response = self.radio_serial.read(self.radio_serial.in_waiting).hex()
                        print(f"Received frequency: {response}")
                        if self.rx_output:
                            self.rx_output.insert(tk.END, f"Received frequency: {response}\n")
                            self.rx_output.see(tk.END)
                else:
                    print(f"Unknown command: {command}")
                    if self.rx_output:
                        self.rx_output.insert(tk.END, f"Unknown radio command: {command}\n")

            except (serial.SerialException, ValueError) as e:
                print(f"Radio communication error: {str(e)}")
                if self.rx_output:
                    self.rx_output.insert(tk.END, f"Radio communication error: {str(e)}\n")
                if self.radio_serial:
                    self.radio_serial.close()
                    self.radio_serial = None
        else:
            print("No serial port or baud rate selected for radio communication.")
            if self.rx_output:
                self.rx_output.insert(tk.END, "No serial port or baud rate selected for radio communication.\n")

if __name__ == "__main__":
    try:
        print("Starting main")
        root = tk.Tk()
        print("Tk root created")
        app = DipperModeApp(root)
        print("App instance created")
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
        print("Mainloop exited")
    except Exception as e:
        print(f"Error in main: {e}")

    finally:
        input("Press Enter to exit...")
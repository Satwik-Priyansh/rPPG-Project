import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
# Import your CNNrPPG model and utils here
# from model import CNNrPPGc
# import utils

class RPPGApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Pulse Detection")
        self.setGeometry(100, 100, 1200, 600)

        # Create main widget and layout
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Create left panel for video feed and BPM
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(640, 480)
        left_layout.addWidget(self.video_label)

        self.bpm_label = QLabel("BPM: --", self)
        self.bpm_label.setAlignment(Qt.AlignCenter)
        self.bpm_label.setFont(QFont("Arial", 24, QFont.Bold))
        left_layout.addWidget(self.bpm_label)

        main_layout.addWidget(left_panel)

        # Create right panel for pulse signal plot
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.figure, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start", self)
        self.stop_button = QPushButton("Stop", self)
        self.start_button.clicked.connect(self.start_processing)
        self.stop_button.clicked.connect(self.stop_processing)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        right_layout.addLayout(button_layout)

        main_layout.addWidget(right_panel)

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)

        # Initialize CNNrPPG model
        # self.model = CNNrPPG()
        # self.model.eval()

        # Variables for processing
        self.is_running = False
        self.pulse_signals = []
        self.bpm_values = []

        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Create a timer for updating frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def start_processing(self):
        self.is_running = True
        self.timer.start(33)  # ~30 fps
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_processing(self):
        self.is_running = False
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Display the frame
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

            if self.is_running:
                # Perform rPPG processing here
                # You would normally use your CNNrPPG model and process the frame

                # For demonstration, we'll use a smoothed random value
                pulse_value = 0.8 * (self.pulse_signals[-1] if self.pulse_signals else 0) + 0.2 * np.random.random()
                self.pulse_signals.append(pulse_value)

                # Keep only the last 100 values
                self.pulse_signals = self.pulse_signals[-100:]

                # Update BPM (smoothed random value for demonstration)
                new_bpm = 0.9 * (self.bpm_values[-1] if self.bpm_values else 70) + 0.1 * np.random.randint(60, 100)
                self.bpm_values.append(new_bpm)
                self.bpm_values = self.bpm_values[-10:]  # Keep last 10 values for smoothing
                avg_bpm = sum(self.bpm_values) / len(self.bpm_values)
                self.bpm_label.setText(f"BPM: {avg_bpm:.0f}")

                # Update pulse signal plot
                self.ax.clear()
                self.ax.plot(self.pulse_signals, color='red', linewidth=2)
                self.ax.set_ylim(0, 1)
                self.ax.set_title("Pulse Signal")
                self.ax.set_facecolor('#f0f0f0')
                self.figure.patch.set_facecolor('#f0f0f0')
                self.canvas.draw()

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RPPGApp()
    window.show()
    sys.exit(app.exec_())
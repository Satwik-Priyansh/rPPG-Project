import cv2
import torch
import numpy as np  # Make sure numpy is imported
from model import CNNrPPG
import utils

def rppg_webcam_cnn():
    cap = cv2.VideoCapture(0)
    model = CNNrPPG()
    model.eval()
    fs = 30  # Assuming 30 frames per second (fps)

    # List to store pulse signal over time
    pulse_signals = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Extract ROI (e.g., the forehead)
        roi = frame[100:200, 150:250]
        cv2.rectangle(frame, (150, 100), (250, 200), (0, 255, 0), 2)

        # Preprocess ROI for CNN input
        roi_resized = cv2.resize(roi, (224, 224))
        roi_tensor = torch.from_numpy(roi_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Get pulse signal prediction from CNN
        with torch.no_grad():
            pulse_signal = model(roi_tensor).numpy().flatten()

        # Add pulse signal to the list
        pulse_signals.append(pulse_signal)

        # Ensure we have enough data points to apply the bandpass filter
        if len(pulse_signals) > 33:
            pulse_signals = pulse_signals[-100:]  # Keep only the last 100 frames for processing

            # Flatten the list and apply bandpass filter
            pulse_signals_np = np.concatenate(pulse_signals).ravel()  # Convert list to NumPy array
            filtered_pulse = utils.bandpass_filter(pulse_signals_np, fs=fs)

            # Calculate bpm
            bpm = utils.calculate_bpm(filtered_pulse, fs=fs)

            # Display the bpm on the frame
            cv2.putText(frame, f"BPM: {bpm:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow('rPPG - CNN Method', frame)

        # Exit loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

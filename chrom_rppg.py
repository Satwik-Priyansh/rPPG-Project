import cv2
import numpy as np
import utils

def rppg_webcam_chrom():
    cap = cv2.VideoCapture(0)
    rgb_signals = []
    fs = 30  # Assuming 30 frames per second (fps)
    alpha = 0.77
    beta = 0.51

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Extract ROI (e.g., the forehead)
        roi = frame[100:200, 150:250]  # Modify these coordinates as per the subject's face
        cv2.rectangle(frame, (150, 100), (250, 200), (0, 255, 0), 2)

        # Extract mean RGB values from ROI
        mean_rgb = np.mean(roi, axis=(0, 1))
        norm_rgb = mean_rgb / np.linalg.norm(mean_rgb)

        # Compute chrominance signals (X and Y)
        X_chrom = norm_rgb[0] - norm_rgb[1]  # R - G
        Y_chrom = alpha * norm_rgb[0] + beta * norm_rgb[1] - (alpha + beta) * norm_rgb[2]  # αR + βG - (α + β)B

        rgb_signals.append([X_chrom, Y_chrom])

        if len(rgb_signals) > 33:
            rgb_signals = rgb_signals[-100:]  # Keep only the last 100 frames

            # Combine chrominance signals into a pulse signal
            rgb_signals_np = np.array(rgb_signals)
            pulse_signal = rgb_signals_np[:, 0] + rgb_signals_np[:, 1]

            # Apply bandpass filter and calculate bpm
            filtered_pulse = utils.bandpass_filter(pulse_signal, fs=fs)
            bpm = utils.calculate_bpm(filtered_pulse, fs=fs)

            # Display the bpm on the frame
            cv2.putText(frame, f"BPM: {bpm:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow('rPPG - CHROM Method', frame)

        # Exit loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

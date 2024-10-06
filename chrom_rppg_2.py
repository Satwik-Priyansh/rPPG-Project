import cv2
import numpy as np
import utils
import matplotlib.pyplot as plt

def rppg_webcam_chrom():
    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    rgb_signals = []
    fs = 30  # Assuming 30 frames per second (fps)
    alpha = 0.77
    beta = 0.51

    plt.ion()  # Turn on interactive mode for real-time plotting
    fig, ax = plt.subplots()
    ax.set_ylim(-1, 1)  # Set y-axis limits for the pulse signal
    line, = ax.plot([], [])
    ax.set_title('Real-Time Pulse Signal')
    ax.set_xlabel('Frames')
    ax.set_ylabel('Pulse Signal')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) > 0:
            # Take the first detected face
            (x, y, w, h) = faces[0]

            # Define the region of interest (ROI) as the forehead area
            roi = frame[y:y + h // 3, x:x + w]

            # Draw a rectangle around the face and the ROI on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h // 3), (0, 255, 0), 2)

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

                # Update the plot for the pulse signal
                line.set_xdata(np.arange(len(filtered_pulse)))
                line.set_ydata(filtered_pulse)
                ax.set_xlim(0, len(filtered_pulse))  # Set x-axis limit to the number of frames
                plt.pause(0.01)  # Pause to allow the plot to update

                # Display the bpm on the frame
                cv2.putText(frame, f"BPM: {bpm:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow('rPPG - CHROM Method', frame)

        # Exit loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    plt.close(fig)  # Close the plot window
    cap.release()
    cv2.destroyAllWindows()

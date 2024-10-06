from scipy.signal import butter, filtfilt
from scipy.fftpack import fft
import numpy as np

def bandpass_filter(signal, fs, lowcut=0.7, highcut=4.0, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y

def calculate_bpm(pulse_signal, fs):
    # Apply FFT to the pulse signal to extract the frequency components
    N = len(pulse_signal)
    freqs = np.fft.fftfreq(N, d=1/fs)
    fft_values = np.abs(fft(pulse_signal))

    # Consider only the positive half of frequencies (real signal)
    positive_freqs = freqs[:N//2]
    positive_fft_values = fft_values[:N//2]

    # Find the dominant frequency
    dominant_freq_idx = np.argmax(positive_fft_values)
    dominant_freq_hz = positive_freqs[dominant_freq_idx]

    # Convert frequency (in Hz) to beats per minute (bpm)
    bpm = dominant_freq_hz * 60
    return bpm

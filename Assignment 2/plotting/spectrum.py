import librosa
import numpy as np
import matplotlib.pyplot as plt


def plot_spectrum(y, sr, start_time, n_fft):
    y_segment = y[int(start_time * sr) : int((start_time + 1) * sr)]
    X = np.abs(librosa.stft(y_segment, n_fft=n_fft))

    freqs = np.fft.fftfreq(n_fft, 1 / sr)
    positive_freqs = freqs[: n_fft // 2]
    spectrum = np.mean(X[: n_fft // 2], axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(positive_freqs, spectrum)
    plt.title(f"Spectrum (start_time={start_time}s, N={n_fft})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()


audio_file = "../assets/aa.ogg"
y, sr = librosa.load(audio_file)

plot_spectrum(y, sr, 0, 1024)

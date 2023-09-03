import librosa
import numpy as np
import matplotlib.pyplot as plt


def plot_spectrogram(y, sr, window_size, hop_length, n_fft):
    spectogram = np.abs(
        librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=window_size)
    )
    spectogram_db = librosa.amplitude_to_db(spectogram, ref=np.max)

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(
        spectogram_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram (window={window_size}s, hop={hop_length}s, N={n_fft})")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()


audio_file = "../assets/aa.ogg"
y, sr = librosa.load(audio_file)

window_sizes = [1024, 2048, 4096]
hop_lengths = [512, 1024, 2048]
n_ffts = [1024, 2048, 4096]

plot_spectrogram(y, sr, 1024, 512, 4096)

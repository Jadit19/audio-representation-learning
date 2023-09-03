import librosa
import numpy as np
import matplotlib.pyplot as plt


def plot_mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(
        mel_spec_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Mel Frequency (Hz)")
    plt.show()


def plot_mel_spectrum(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_frequencies = librosa.mel_frequencies(n_mels=n_mels)
    mel_spectrum = np.sum(mel_spec_db, axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(mel_frequencies, mel_spectrum)
    plt.title("Mel Spectrum (dB)")
    plt.xlabel("Mel Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.show()


audio_file = "../assets/aa.ogg"
y, sr = librosa.load(audio_file)

plot_mel_spectrogram(y, sr)
plot_mel_spectrum(y, sr)

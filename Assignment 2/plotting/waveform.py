import librosa
import numpy as np
import matplotlib.pyplot as plt

audio_file = "../assets/sound.ogg"
y, sr = librosa.load(audio_file)

time = np.arange(0, len(y)) / sr

plt.figure(figsize=(10, 4))
plt.plot(time, y)
plt.title("Waveform of Audio")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

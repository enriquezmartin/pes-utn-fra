import librosa
import librosa.display
import numpy as np
import scipy.signal as signal
import sounddevice as sd
import matplotlib.pyplot as plt
from tkinter import filedialog

# Función para cambiar el pitch
def change_pitch(audio, sr, semitones):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)

# Función para aplicar un filtro paso-bajo
def apply_lowpass(audio, sr, cutoff_freq):
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype='low') # Filtro de Butterworth 
    return signal.filtfilt(b, a, audio)

# Cargar el audio
# filename = "C:/Users/marti/Documents/Grabaciones de sonido/robaresmalo.wav"  # Reemplaza con tu archivo
filename = filedialog.askopenfilename(title="Elige un archivo .wav",
                                      initialdir="C:/Users/marti/Documents/Grabaciones de sonido/",
                                      filetypes=[("Waveform files", "*.wav"), ("All files", "*.*")])
audio, sr = librosa.load(filename, sr=None)

# Modificar pitch
semitones = 4  # Cambiar a +4 semitonos
audio_pitch_changed = change_pitch(audio, sr, semitones)

# Aplicar filtro pasa-bajos
cutoff_freq = 600  # Frecuencia de corte en Hz
audio_filtered = apply_lowpass(audio, sr, cutoff_freq)

# Reproducir audios
print("Reproduciendo audio original...")
sd.play(audio, sr)
sd.wait()

print("Reproduciendo audio con pitch modificado...")
sd.play(audio_pitch_changed, sr)
sd.wait()

print("Reproduciendo audio con filtro pasa-bajos aplicado...")
sd.play(audio_filtered, sr)
sd.wait()

# Graficar espectros
fig = plt.figure(figsize=(15, 10))
fig.canvas.manager.set_window_title("Gráficos de las formas de onda")
# Espectro del audio original
plt.subplot(3, 1, 1)
librosa.display.waveshow(audio, sr=sr, alpha=0.5)
plt.title("Audio Original")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")

# Espectro del audio con pitch cambiado
plt.subplot(3, 1, 2)
librosa.display.waveshow(audio_pitch_changed, sr=sr, alpha=0.5)
plt.title("Audio con Pitch Modificado (+4 semitonos)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")

# Espectro del audio filtrado
plt.subplot(3, 1, 3)
librosa.display.waveshow(audio_filtered, sr=sr, alpha=0.5)
plt.title("Audio Filtrado (Pasa-bajos a 300 Hz)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")

plt.tight_layout()
plt.show()

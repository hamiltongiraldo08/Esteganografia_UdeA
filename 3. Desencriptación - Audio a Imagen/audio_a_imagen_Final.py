import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, windows
from scipy.io import wavfile

# Cargar el archivo WAV
fs, audio = wavfile.read("pilares-creacion-2022_lames-webb.jpg.wav")

# Convertir el audio a formato float y normalizar
if audio.dtype != np.float32 and audio.dtype != np.float64:
    audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

# Definir la longitud de la ventana para el analisis para la STF
w_length = 1024
	
# Aplicar la STFT
f, t, image_re = stft(audio, fs, window = windows.kaiser(w_length, beta = 1),
                     nperseg = w_length, noverlap = 0, nfft = 1024)

# Mostrar la imagen reconstruida desde su audio
plt.figure(figsize = (5, 5))
plt.imshow(np.flipud(np.abs(image_re)), cmap = 'gray', aspect = 'auto')
plt.title("Imagen reconstruida desde su audio")
plt.axis("off")
plt.tight_layout()
plt.show()

'''Audio Steganography Project in MATLAB
https://github.com/hitmansixteen/Audio-Steganography-1/blob/main/README.md'''

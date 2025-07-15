import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.io import wavfile
from numpy.fft import fft

# Cargar el archivo WAV de la imagen
Fs, audio_image = wavfile.read("pluto.png.wav")
audio_image = audio_image.astype(float)
if audio_image.ndim > 1:
    audio_image = audio_image[:, 0]

# Cargar el archivo WAV de la cancion
Fs, song = wavfile.read("new_horizons_song.wav")
song = song.astype(np.float32)
if song.ndim > 1:
    song = song[:, 0]

# Sin permutacion
data = np.real(audio_image.flatten())

# Con permutacion
key = 2
w_length = 1024
n = len(audio_image)
rng = np.random.default_rng(key)
p = rng.permutation(n)
data_perm = np.real(audio_image[p])

# Parametros
scaled = 40
n_samples = 3000000
len_data = len(data_perm)

# Funcion para ocultar los datos en la parte de alta frecuencia usando DWT
def encryption_dwt(audio, payload, n_samples, len_data, scale):
    audio = audio[:n_samples]
    cA, cD = pywt.dwt(audio, 'db12')
    steg_old = cD.copy()
    cD_steg = cD.copy()
    cD_steg[:len_data] = payload / scale
    steg_new = cD_steg
    image_song = pywt.idwt(cA, cD_steg, 'db12')
    return image_song, steg_old, steg_new

# Ocultar sin permutar
image_song, _, steg_new = encryption_dwt(
    song, data, n_samples, len_data, scaled)

# Ocultar con permutacion
image_song_perm, steg_old, steg_new_perm = encryption_dwt(
    song, data_perm, n_samples, len_data, scaled)

# Guardar con permutacion
image_song_norm = image_song_perm / np.max(np.abs(image_song))
image_song_int16 = (image_song_norm * 32767).astype(np.int16)
wavfile.write("esteg_con_per.wav", Fs, image_song_int16)

# Guardar sin permutacion
image_song_norm = image_song / np.max(np.abs(image_song))
image_song_int16 = (image_song_norm * 32767).astype(np.int16)
wavfile.write("esteg_sin_per.wav", Fs, image_song_int16)

print("se guardaron los archivos esteg_con_per.wav y esteg_sin_per.wav")

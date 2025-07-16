import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.io import wavfile
from scipy.signal import stft, windows

# Parametros
w_length = 1024
Fs = 44100
scaled = 20
key_correct = 2
key_wrong = 8

# Cargar el audio con esteganografia
audio_path = 'esteg_sin_per.wav'
Fs, image_song_perm = wavfile.read(audio_path)

# Convertir a mono si es estéreo
if image_song_perm.ndim > 1:
    image_song_perm = image_song_perm[:, 0]

n_samples = 3000000
if len(image_song_perm) < n_samples:
    n_samples = len(image_song_perm)

len_data = int((w_length / 2) * (w_length / 2) * 2)
image_song_perm = image_song_perm.astype(np.float32)


def decryption_dwt(signal, n_samples, len_data, scale):
    _, cD = pywt.dwt(signal[:n_samples], 'db12')
    recovered = cD[:len_data] * scale
    return recovered


def stft_image(signal, fs, w_length):
    f, t, image_re = stft(signal, fs, window=windows.kaiser(w_length, beta=1),
                          nperseg=w_length, noverlap=0, nfft=1024)
    return np.abs((image_re))


image_re = decryption_dwt(image_song_perm, n_samples, len_data, scaled)

# Sin clave
no_key = image_re[:len_data]
image_no_key = stft_image(no_key, Fs, w_length)

# Con clave incorrecta
rng = np.random.default_rng(key_wrong)
p_wrong = rng.permutation(len_data)
extract_wrong_key = np.zeros_like(no_key)
extract_wrong_key[p_wrong] = image_re[:len(p_wrong)]
image_wrong_key = stft_image(extract_wrong_key, Fs, w_length)

# Con clave correcta
rng = np.random.default_rng(key_correct)
p_correct = rng.permutation(len_data)
extract_good_key = np.zeros_like(no_key)
extract_good_key[p_correct] = image_re[:len(p_correct)]
image_good_key = stft_image(extract_good_key, Fs, w_length)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(np.flipud(np.abs(image_no_key)),
           cmap ='gray', aspect ='auto')
plt.title('Imagen recuperada sin llave')

plt.subplot(1, 3, 2)
plt.imshow(image_wrong_key, cmap ='gray', aspect ='auto')
plt.title('Imagen recuperada con la llave incorrecta')

plt.subplot(1, 3, 3)
plt.imshow(np.flipud(np.abs(image_good_key)),
           cmap ='gray', aspect ='auto')
plt.title('Imagen recuperada con la llave correcta')

plt.tight_layout()
plt.show()

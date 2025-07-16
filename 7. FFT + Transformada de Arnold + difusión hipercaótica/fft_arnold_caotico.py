'''En este código se desarrolla un cifrado de imágenes donde se combina
   la Transformada Rápida de Fourier FFT, el mapeo de Arnold y la difusión Hipercaótica'''

import cv2 # OpenCV usado para procesar imágenes
import numpy as np # Operaciones numéricas
import pywt # PyWavelets: Transformadas de Wavelet
from scipy.integrate import odeint # Solucionador de Ecuaciones diferenciales
import matplotlib.pyplot as plt # Graficación

'''Leer imagen'''

def lectura_imagen(image_path):
    # Lectura en escala de grises
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    if img is None:
        # Por si hay error
        raise ValueError("Imagen no localizada o ruta incorrecta") 
    return img

'''Mapeo de Arnold'''

def arnold_map_scramble(image, iterations=2):
    N = image.shape[0] # Para tamaño de imágenes cuadradas
    scrambled = image.copy() # Copia la imagen para dejar intacta la inicial

    for _ in range(iterations): # Ciclo iterativo
        temp = scrambled.copy()
        for y in range(N):
            for x in range(N):
                '''Nuevas coordenadas x,y'''
                new_x = (x + y) % N
                new_y = (x + 2 * y) % N
                '''Reorganización de pixeles'''
                scrambled[new_y, new_x] = temp[y, x]
    return scrambled

'''Desciframiento (Mapeo inverso de Arnold)'''

def inverse_arnold_map_scramble(image, iterations=2):
    N = image.shape[0]
    unscrambled = image.copy()
    for _ in range(iterations):
        temp = unscrambled.copy()
        for y in range(N):
            for x in range(N):
                '''Operaciones inversas para x,y'''
                new_x = (2 * x - y) % N
                new_y = (-x + y) % N
                unscrambled[new_y, new_x] = temp[y, x]
    return unscrambled

'''Sistema Hipercaótico'''

def solve_hyperchaotic_system(num_points, x0=[0.3, -0.2, 1, 1.4], a=36, \
                                                b=3, c=28, d=-16, k=0.2):
    def hyperchaotic_system(X, t):
        x1, x2, x3, x4 = X
        '''Ecuaciones diferenciales'''
        dx1 = a * (x2 - x1)
        dx2 = -x1 * x3 + d * x1 + c * x2 - x4
        dx3 = x1 * x2 - b * x3
        dx4 = x1 + k
        return [dx1, dx2, dx3, dx4]
    
    t = np.linspace(0, 100, num_points) # Linspace de tiempo
    sol = odeint(hyperchaotic_system, x0, t) # Solucionador del sistema
    return sol[:, 0], sol[:, 1], sol[:, 3], sol[:, 2] # Devolución de secuencias

'''Cifrado (Difusión Hipercaótica)'''

def difusion_hipercaotica(image):
    rows, cols = image.shape
    num_pixels = rows * cols
    '''Creación de secuencias caóticas'''
    x1_seq, x2_seq, x3_seq, x4_seq = solve_hyperchaotic_system(num_pixels // 4)
    
    # Procesamiento de secuencias caóticas. 
    x1_seq = np.uint8(np.mod(np.floor((np.abs(x1_seq) \
                    - np.floor(np.abs(x1_seq))) * 1e14), rows))
    x2_seq = np.uint8(np.mod(np.floor((np.abs(x2_seq) \
                    - np.floor(np.abs(x2_seq))) * 1e14), rows))
    x3_seq = np.uint8(np.mod(np.floor((np.abs(x3_seq) \
                    - np.floor(np.abs(x3_seq))) * 1e14), rows))
    x4_seq = np.uint8(np.mod(np.floor((np.abs(x4_seq) \
                    - np.floor(np.abs(x4_seq))) * 1e14), rows))
    
    img_vector = image.flatten() # Conversión de imagen a vector
    C = np.zeros_like(img_vector)
    C[0] = 251 # Valor de inicio
    
    # Aplicación de difusión XOR para secuencias caóticas
    for i in range(0, num_pixels, 4):
        idx = i // 4
        if i + 3 < num_pixels:
            C[i] = img_vector[i] ^ x1_seq[idx] ^ C[i - 1 if i > 0 else 0]
            C[i + 1] = img_vector[i + 1] ^ x2_seq[idx] ^ C[i]
            C[i + 2] = img_vector[i + 2] ^ x3_seq[idx] ^ C[i + 1]
            C[i + 3] = img_vector[i + 3] ^ x4_seq[idx] ^ C[i + 2]
    
    # Crea nuevamente la imagen (la forma nuevamente).
    diffused_img = C.reshape(rows, cols) 
    return diffused_img

'''Descifrado (Difusión hipercaótica inversa)'''

def difusion_hipercaotica_inversa(encrypted_img):
    '''Forma inversa al difusion_hipercaotica'''
    rows, cols = encrypted_img.shape
    num_pixels = rows * cols
    x1_seq, x2_seq, x3_seq, x4_seq = solve_hyperchaotic_system(num_pixels // 4)
    
    # Procesamiento de las secuencias caóticas
    x1_seq = np.uint8(np.mod(np.floor((np.abs(x1_seq) \
                    - np.floor(np.abs(x1_seq))) * 1e14), rows))
    x2_seq = np.uint8(np.mod(np.floor((np.abs(x2_seq) \
                    - np.floor(np.abs(x2_seq))) * 1e14), rows))
    x3_seq = np.uint8(np.mod(np.floor((np.abs(x3_seq) \
                    - np.floor(np.abs(x3_seq))) * 1e14), rows))
    x4_seq = np.uint8(np.mod(np.floor((np.abs(x4_seq) \
                    - np.floor(np.abs(x4_seq))) * 1e14), rows))
    
    enc_vector = encrypted_img.flatten()
    C = np.zeros_like(enc_vector)
    C[0] = 251
    
    for i in range(num_pixels - 1, 3, -4): # Operaciones XOR en orden inverso
        idx = i // 4
        enc_vector[i] ^= x4_seq[idx] ^ enc_vector[i - 1]
        enc_vector[i - 1] ^= x3_seq[idx] ^ enc_vector[i - 2]
        enc_vector[i - 2] ^= x2_seq[idx] ^ enc_vector[i - 3]
        enc_vector[i - 3] ^= x1_seq[idx] ^ (C[0] if i - 3 == 0 else enc_vector[i - 4])
    
    original_img = enc_vector.reshape(rows, cols)
    return original_img

'''Cifrado con FFT y Arnold'''

def fft_scramble(image):
    # 1. Aplicar FFT y centrar frecuencias
    fft_img = np.fft.fftshift(np.fft.fft2(image))
    
    # 2. Separar magnitudes y fases
    magnitudes = np.abs(fft_img)
    phases = np.angle(fft_img)
    
    # 3. Revolver solo las magnitudes con Arnold
    scrambled_magnitudes = arnold_map_scramble(magnitudes)
    
    # 4. Reconstruir FFT y aplicar inversa
    scrambled_fft = scrambled_magnitudes * np.exp(1j * phases)
    scrambled_img = np.fft.ifft2(np.fft.ifftshift(scrambled_fft)).real
    
    # Asegurar valores válidos de píxeles
    return np.clip(scrambled_img, 0, 255).astype(np.uint8) 

'''Descifrado con FFT y Arnold'''

def fft_unscramble(scrambled_image):
    # 1. Aplicar FFT a la mezcla
    fft_scrambled = np.fft.fftshift(np.fft.fft2(scrambled_image))
    
    # 2. Separar magnitudes y fases
    scrambled_magnitudes = np.abs(fft_scrambled)
    phases = np.angle(fft_scrambled)
    
    # 3. Devolver (proceso de devolver - revertir) Arnold en magnitudes
    unscrambled_magnitudes = inverse_arnold_map_scramble(scrambled_magnitudes)
    
    # 4. Reconstruir imagen original
    unscrambled_fft = unscrambled_magnitudes * np.exp(1j * phases)
    original_img = np.fft.ifft2(np.fft.ifftshift(unscrambled_fft)).real
    
    return np.clip(original_img, 0, 255).astype(np.uint8)

'''Cifrado y Descifrado completo'''

'''Cifrado FFT + Arnold + Difusión Caótica'''
def chaotic_image_encryption_fft(img):
    # Paso 1: Revolver con FFT + Arnold (solo en dominio de frecuencia)
    scrambled_img = fft_scramble(img)
    
    # Paso 2: Difusión caótica (en dominio espacial)
    encrypted_img = difusion_hipercaotica(scrambled_img)
    return encrypted_img

'''Descifrado completo'''
def chaotic_image_decryption_fft(encrypted_img):
    # Paso 1: Revertir difusión
    scrambled_img = difusion_hipercaotica_inversa(encrypted_img)
    
    # Paso 2: Revertir FFT + Arnold
    decrypted_img = fft_unscramble(scrambled_img)
    return decrypted_img

'''Programa Principal y visualización'''

if __name__ == "__main__":
    # Cargar imagen
    img = lectura_imagen("saturn.jpg")
    
    # Probar FFT + Arnold
    fft_arnold_scrambled = fft_scramble(img)
    fft_arnold_unscrambled = fft_unscramble(fft_arnold_scrambled)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    
    plt.subplot(1, 3, 2)
    plt.imshow(fft_arnold_scrambled, cmap='gray')
    plt.title('FFT + Arnold')
    
    plt.subplot(1, 3, 3)
    plt.imshow(fft_arnold_unscrambled, cmap='gray')
    plt.title('Descifrado FFT')
    plt.show()
    
    # Cifrado/Descifrado completo con FFT
    encrypted_img = chaotic_image_encryption_fft(img)
    decrypted_img = chaotic_image_decryption_fft(encrypted_img)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    
    plt.subplot(1, 3, 2)
    plt.imshow(encrypted_img, cmap='gray')
    plt.title('Cifrado (FFT + Arnold + Difusión hipercaótica)')
    
    plt.subplot(1, 3, 3)
    plt.imshow(decrypted_img, cmap='gray')
    plt.title('Descifrado')
    plt.show()

'''
El código base de Matlab del que se partió para realizar este, fue obtenido del desarrollo realizado por:
Huiben Zhang, Shi Min Liu, Min Gao and Mengmeng Zhang, "Chaotic image encryption algorithm 
research based on Contourlet transformation," 2015 12th International Computer Conference 
on Wavelet Active Media Technology and Information Processing (ICCWAMTIP), Chengdu, 2015, 
pp. 303-306, doi: 10.1109/ICCWAMTIP.2015.7493997.
https://github.com/cgurkan/image-encryption-using-chaotic-map.git
'''
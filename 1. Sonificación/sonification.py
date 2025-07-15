"""Sonificación de Imágenes"""

""" Convertir una imagen en señal de audio
    para visualizarse con un espectrograma
"""

import array
import struct
import sys
import wave  # Permitir crear y manipular archivos de audio WAV

# Permitir cargar y manipular imágenes
from PIL import Image

# Transformada inversa de Fourier. 
# Para transformar datos del dominio de la frecuencia a dominio del tiempo (audio)
from numpy.fft import ifft


"""Configuración Inicial"""

# Ruta de la Imagen que se va a convertir
rutaimagen = r"C:\Users\USUARIO\Downloads\ensayo.jpg"

# Ancho deseado, 512 píxeles para redimensionar la imagen
ancho_deseado = 512

# Frecuencia de muestreo para el audio de salida en Hz
frec_muestreo = 44100.0


"""Función para cargar y convertir la imagen"""

""" Abre la imagen y la convierte a escala de grises, 
    calcula la proporción 'ratio' para mantener la relación de aspecto
    y redimensiona la imagen para que tenga 'ancho_deseado' de ancho.
"""
def cargar_imagen(archivo):
    """Conversión a escala de grises mediante 'L' """
    im = Image.open(archivo).convert("L")
    ancho, altura = im.size
    ratio = ancho_deseado / ancho
    return im.resize((ancho_deseado, int(altura * ratio)))


""" Procesamiento de la imagen, carga y relleno"""

# Imagen redimensionada y en grises
im = cargar_imagen(rutaimagen)

"""Lista de ceros para rellenar cada columna antes de aplicar la IFFT. 
Esto asegura que cada vector tenga el doble de tamaño 
(para tener 1024 puntos si ancho_deseado es 512).
"""
pad = [0] * ancho_deseado


"""Función para acceder a un píxel específico"""
def pix(im, x, y):
    """Obtiene el valor de un píxel"""
    return im.getpixel((x, y))  # Retorna la intensidad de gris del píxel en la 
                                # posición (x, y).


"""Lista para almacenar los resultados"""
# Contiene los resultados de la transformada inversa (IFFT) por cada columna de 
# píxeles.
resultados = []


"""Transformación de la imagen a audio"""

""" Para cada columna de la imagen: Extrae sus valores de gris 
    (de abajo hacia arriba, con insert(0, pix(im, i, j))).

Rellena con ceros (pad) y aplica la IFFT.

Guarda el resultado.
"""
def transform(im):
    """Aplica la transformada de Fourier a los píxeles de la imagen"""
    ancho, altura = im.size
    for i in range(ancho):  # para cada columna
        line = []
        for j in range(altura):  # de arriba hacia abajo
            line.insert(0, pix(im, i, j))  # invierte el orden vertical
        result = ifft(line + pad)  # aplica IFFT a los datos extendidos
        resultados.append(result)


""" Transformación y cálculo de parámetros"""

"""Cálculo de duración de la señal"""
transform(im)
# Total de muestras de audio = columnas × longitud de cada IFFT.
total_muestras = len(resultados[0]) * len(resultados)  
# Se imprime la duración total de la señal de audio en segundos.
print(f"Duración (s): {total_muestras / frec_muestreo}")  


""" Configuración del archivo WAV"""

# Abre un archivo WAV para escritura
w = wave.open(f"{rutaimagen}.wav", "w")

# Se establece: 1 canal (mono), 2 bytes por muestra (16 bits), frecuencia de muestreo,
# número total de muestras, sin compresión "NONE"
w.setparams((1, 2, frec_muestreo, total_muestras, "NONE", ""))


"""Normalización de amplitudes"""
# Encuentra la mayor amplitud real en todos los resultados para escalar los valores de
# forma que no saturen el audio.
mayor_amplitud = max(abs(x.real) for row in resultados for x in row)
print(f"Mayor Amplitud: {mayor_amplitud}")


"""Escalado y escritura del archivo"""

"""Para cada línea (columna de la imagen), se crea un buffer vacío, 
    cada valor real se escala y se convierte a formato binario de 16 bits 
    (h = signed short).
    Se escribe la línea convertida al archivo de audio
"""
bigint = 2 ** 14  # para convertir a 16-bit: ±32768

# Escritura de los datos de audio
for line in resultados:
    buf = b""
    for i in line:
        val = i.real
        # Escalar y convertir a binario
        buf += struct.pack("h", int(val * (bigint / mayor_amplitud)))  
    w.writeframes(buf)


"""Cerrar el audio"""
# Finaliza la escritura del archivo WAV
w.close()


""" Traducido a python desde el Código base .hy  tomado de github Bryan Garza, Spectre
    Convert images to audio, visible with a spectrograph. 
    Uses Inverse Fast Fourier Transform.
"""
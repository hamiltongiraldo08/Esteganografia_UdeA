"""Esteganografía con Fourier"""


"""Librerías"""
import math  # Funciones matemáticas
import struct  # Para empaquetamiento de datos binarios
import time  # Marcas de tiempo
import wave  # Leer y escribir archivos de audio WAV
import numpy as np  # Manipular matrices
import scipy.ndimage  # Redimensionamiento de fotos
from PIL import Image  # Procesar imágenes luego de cargarlas
from pydub import AudioSegment  # Mezcla pistas de audio
from tqdm import tqdm  # Mostrar progreso de ejecución


""" Función para cargar una imagen y convertirla a una matriz normalizada 
    en escala de grises"""
def cargar_imagen(file):
    img = Image.open(file)  # Abre el archivo
    img = img.convert('L')  # Convierte la imagen a escala de grises

    img_arr = np.array(img)  # Convierte la imagen a un arreglo NumPy
    img_arr = np.flip(img_arr, axis=0)  # Invierte verticalmente la imagen

    img_arr -= np.min(img_arr)  # Resta el valor mínimo para normalizar
    # Divide entre el máximo para normalizar entre 0 y 1
    img_arr = img_arr / np.max(img_arr)  

    return img_arr  # Regresa la imagen como matriz normalizada


"""Redimensionamiento de la imagen a un nuevo tamaño"""
def redimen_imagen(img_arr, size):
    if not size[0]:
        size[0] = img_arr.shape[0]
    if not size[1]:
        size[1] = img_arr.shape[1]

    # Calcula el factor de escalamiento
    factor_escala = size[0] / img_arr.shape[0], size[1] / img_arr.shape[1]

    # Redimensiona usando interpolación de vecinos más cercanos
    img_arr = scipy.ndimage.zoom(img_arr, factor_escala, order=0)

    return img_arr  # Retorna la imagen redimensionada


""" Función para aplicar un filtro opcional a la imagen, pero actualmente 
    no realiza nada"""
def preprocess_image(img_arr):
    # Filtro inverso
    # img_arr = 1 - img_arr
    return img_arr


"""Función principal que convierte una imagen en una onda de sonido"""
def onda_sonido(file, output='sonido.wav', duracion=2.5, freq_muestreo=44100.0, \
                 min_freq=0, max_freq=22000):
    # Crea un archivo WAV de salida
    waveform = wave.open(output, 'w')
    waveform.setnchannels(1)  # Canal mono
    waveform.setsampwidth(2)  # 2 bytes por muestra, 16 bits
    waveform.setframerate(freq_muestreo)  # Frecuencia de muestreo

    # Total de muestras de audio a generar
    total_muestras_audio = int(duracion * freq_muestreo)  
    max_intensidad = 32768  # Valor máximo para audio de 16 bits

    paso_size = 100  # Paso en el espectro de frecuencias (resolución vertical)
    subpaso_size = 250  # Subpasos dentro de cada rango de frecuencia (para precisión)

    freq_rango = max_freq - min_freq
    stepping_spectrum = int(freq_rango / paso_size)  # Número de bandas de frecuencia

    # Procesamiento de la imagen
    img_arr = cargar_imagen(file)
    img_arr = preprocess_image(img_arr)
    img_arr = redimen_imagen(img_arr, size=(stepping_spectrum, total_muestras_audio))
    img_arr *= max_intensidad  # Escala la intensidad al rango del audio WAV

    # Recorre cada frame de tiempo (columna de la imagen)
    for frame in tqdm(range(total_muestras_audio)):
        signal_val, count = 0, 0  # Inicializa el valor de la señal y el contador

        # Recorre cada banda de frecuencia (fila de la imagen)
        for step in range(stepping_spectrum):
            # Intensidad del píxel en esa banda y tiempo
            intensity = img_arr[step, frame]  

            actual_freq = (step * paso_size) + min_freq  # Frecuencia de inicio
            proxima_freq = ((step + 1) * paso_size) + min_freq  # Frecuencia de fin

            if proxima_freq - min_freq > max_freq:  # Límite del espectro
                proxima_freq = max_freq

            # Genera componentes sinusoidales para este rango
            for freq in range(actual_freq, proxima_freq, subpaso_size):
                signal_val += intensity * math.cos(2 * math.pi * freq * frame / freq_muestreo)
                count += 1

        if count == 0:
            count = 1  # Evita división por cero
        signal_val /= count  # Promedia las componentes

        # Empaqueta el valor como entero de 16 bits y lo escribe
        data = struct.pack('<h', int(signal_val))
        waveform.writeframesraw(data)

    waveform.writeframes(''.encode())  # Finaliza el archivo WAV
    waveform.close()

# Función para mezclar dos pistas de audio (superposición)
def mix_pistas_audio(file1, file2, output_file, start=0):
    sonido1 = AudioSegment.from_file(file1)  # Carga el primer archivo de audio
    sonido2 = AudioSegment.from_file(file2)  # Carga el segundo archivo

    # Superpone sonido2 sobre sonido1 a partir del tiempo "5" segundos
    output = sonido1.overlay(sonido2, position=5000)

    # Exporta el resultado como nuevo archivo
    output.export(output_file, format=output_file.split('.')[-1])

# Función principal del programa
def principal():
    # Archivos de entrada
    # Imagen que se convertirá en sonido
    file = r"C:\Users\USUARIO\Downloads\carinanebula.jpg"  
    # Música que se mezclará con el sonido generado
    music = r"C:\Users\USUARIO\Downloads\himno.mp3"  
    # Archivos de salida (se les agrega una marca de tiempo para evitar sobreescritura)
    nombre_base = str(int(time.time()))[-5:]
    img_sonido_output = f"C:\\Users\\USUARIO\\Downloads\\sonido_imagen_{nombre_base}.wav"
    final_output = f"C:\\Users\\USUARIO\\Downloads\\mezcla_final_{nombre_base}.wav"

    # Genera el sonido a partir de la imagen
    onda_sonido(
        file,
        output=img_sonido_output,
        duracion=10.5,
        freq_muestreo=44100.0,
        min_freq=16000,
        max_freq=22000
    )

    print("La conversión de imagen a sonido finalizó exitosamente")

    # Mezcla la música con el sonido generado
    mix_pistas_audio(music, img_sonido_output, final_output)
    print("La mezcla de sonido finalizó exitosamente")
    print("Proceso completado.")

# Ejecuta el programa si se llama directamente
if __name__ == "__main__":
    principal()



""" Código base Original Creado por Samuel Prevost en 20/12/2018.
    Sus derechos de autor se preservan y se dan los créditos por 
    ser la base de esta edición.
    Código base tomado de su github usr-ein/SpectroGenV2 """
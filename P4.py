#!/usr/bin/env python3

# Copyright (C) 2021 Andrés B.S.
# SPDX-License-Identifier: MIT

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time


def fuente_info(imagen):
    '''Una función que simula una fuente de
    información al importar una imagen y
    retornar un vector de NumPy con las
    dimensiones de la imagen, incluidos los
    canales RGB: alto x largo x 3 canales

    :param imagen: Una imagen en formato JPG
    :return: un vector de pixeles
    '''

    img = Image.open(imagen)

    return np.array(img)


def rgb_a_bits(array_imagen):
    '''Convierte los pixeles de base
    decimal (de 0 a 255) a binaria
    (de 00000000 a 11111111).

    :param imagen: array de una imagen
    :return: Un vector de (1 x k) bits 'int'
    '''
    # Obtener las dimensiones de la imagen
    x, y, z = array_imagen.shape

    # Número total de elementos (pixeles x canales)
    n_elementos = x * y * z

    # Convertir la imagen a un vector unidimensional de n_elementos
    pixeles = np.reshape(array_imagen, n_elementos)

    # Convertir los canales a base 2
    bits = [format(pixel, '08b') for pixel in pixeles]
    bits_Rx = np.array(list(''.join(bits)))

    return bits_Rx.astype(int)


def bits_a_rgb(bits_Rx, dimensiones):
    '''Un blque que decodifica el los bits
    recuperados en el proceso de demodulación

    :param: Un vector de bits 1 x k
    :param dimensiones: Tupla con dimensiones de la img.
    :return: Un array con los pixeles reconstruidos
    '''
    # Cantidad de bits
    N = len(bits_Rx)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)


def canal_ruidoso(senal_Tx, Pm, SNR):
    '''Un bloque que simula un medio de trans-
    misión no ideal (ruidoso) empleando ruido
    AWGN. Pide por parámetro un vector con la
    señal provieniente de un modulador y un
    valor en decibelios para la relación señal
    a ruido.

    :param senal_Tx: El vector del modulador
    :param Pm: Potencia de la señal modulada
    :param SNR: Relación señal-a-ruido en dB
    :return: La señal modulada al dejar el canal
    '''
    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano (potencia = varianza)
    ruido = np.random.normal(0, np.sqrt(Pn), senal_Tx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senal_Tx + ruido

    return senal_Rx


def modulador(bits, fc, mpp):
    '''Un método que simula el esquema de
    modulación digital 16-QAM.

    :param bits: Vector unidimensional de bits
    :param fc: Frecuencia de la portadora en Hz
    :param mpp: Cantidad de muestras por periodo de onda portadora
    :return: Un vector con la señal modulada
    :return: Un valor con la potencia promedio [W]
    :return: La onda portadora coseno c1(t)
    :return: La onda portadora seno c2(t)
    '''
    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits)  # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora c(t)
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp)  # mpp: muestras por período
    portadora1 = np.cos(2*np.pi*fc*t_periodo)
    portadora2 = np.sin(2*np.pi*fc*t_periodo)

    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp)
    senal_Tx = np.zeros(t_simulacion.shape)

    # 4. Asignar las formas de onda según los bits (16-QAM)
    # Obsérvese: la pérdida de información si len(bits)%4!=0

    # Se establece el contador de muestreos j
    # Note que se almacena todo valor entre j y j+1
    # Por eso j se incrementa de forma diferente a i
    j = 0
    for i in range(0, N, 4):
        ''' Se crea la señal según los valores del 16-QAM:
        b1 = bits[i]; b2 = bits[i+1];
        b3 = bits[i+2]; b4 = bits[i+3]
        La siguiente fórmula se diseñó para compactar el código:
        senal_Tx[i*mpp: (i+1)*mpp] = (-1)**(1+b1) * 3**(1-b2) * portadora1 \
            + (-1)**(b3) * 3**(1-n4) * portadora2
        Con la tabla proporcionada se puede atestar que estos valores
        coinciden con los del 16-QAM.
        '''
        senal_Tx[j*mpp: (j+1)*mpp] = \
            (-1)**(1+bits[i]) * 3**(1-bits[i+1]) * portadora1 \
            + (-1)**(bits[i+2]) * 3**(1-bits[i+3]) * portadora2
        j += 1
        # Aquí es donde se que se duplica el ancho de banda
    # 5. Calcular la potencia promedio de la señal modulada
    P_senal_Tx = 1 / (N*Tc) * np.trapz(pow(senal_Tx, 2), t_simulacion)

    return senal_Tx, P_senal_Tx, portadora1, portadora2


def demodulador(senal_Rx, portadora1, portadora2, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema 16-QAM.

    :param senal_Rx: La señal recibida del canal
    :param portadora: La onda portadora c(t)
    :param mpp: Número de muestras por periodo
    :return: Los bits de la señal demodulada
    '''

    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de bits (símbolos) en transmisión
    N = int(M / mpp)

    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)

    # Vector para la señal demodulada
    senal_demodulada = np.zeros(senal_Rx.shape)

    # Pseudo-energía de un período de la portadora
    # Es = np.sum(portadora * portadora)
    j = 0

    # Demodulación
    for i in range(N):

        if j+4 > N:  # La cantidad datos que se puede recuperar no siempre
            break   # coincide con lo que se puede muestrear

        # Producto interno de dos funciones
        producto1 = senal_Rx[i*mpp: (i+1)*mpp] * portadora1
        producto2 = senal_Rx[i*mpp: (i+1)*mpp] * portadora2
        senal_demodulada[i*mpp: (i+1)*mpp] = producto1 + producto2

        # Criterio de decisión por detección de energía y magnitud
        # SE ASUMIÓ QUE LAS SEÑALES ESTÁN EN FASE
        # En la realidad hay que sincronizarlas
        # Si el valor a poner es cero, no se toca pues ya es cero
        # por defecto; esto es una pequeña optimización

        # Primero se detecta el signo de la amplitud
        if np.sum(producto1) >= 0:
            bits_Rx[j] = 1  # b1

        # Ojo que acá se invierte el signo
        if np.sum(producto2) < 0:
            bits_Rx[j+2] = 1  # b3

        # Ahora se analiza la magnitud (partiendo del
        # punto medio entre 1 y 3+1) 
        if np.max(np.abs(producto1)) < 2.5:
            bits_Rx[j+1] = 1  # b2
        if np.max(np.abs(producto2)) < 2.5:
            bits_Rx[j+3] = 1  # b4
        j += 4
        # DEBUG: print(f"{amp1} {amp2} \n")
    return bits_Rx.astype(int), senal_demodulada


# MAIN
# Parámetros
fc = 5000  # frecuencia de la portadora
mpp = 20   # muestras por periodo de la portadora
SNR = 120   # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx = fuente_info('arenal.jpg')
dimensiones = imagen_Tx.shape

# 2. Codificar los pixeles de la imagen
bits_Tx = rgb_a_bits(imagen_Tx)

# 3. Modular la cadena de bits usando el esquema 16-QAM
senal_Tx, Pm, portadora1, portadora2 = modulador(bits_Tx, fc, mpp)

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

# 5. Se desmodula la señal recibida del canal
bits_Rx, senal_demodulada = demodulador(senal_Rx, portadora1, portadora2, mpp)

# 6. Se visualiza la imagen recibida
imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)
Fig = plt.figure(figsize=(10, 6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular número de errores
errores = sum(abs(bits_Tx - bits_Rx))
BER = errores/len(bits_Tx)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Transmitido')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx)
ax.set_title('Recuperado')
Fig.tight_layout()

plt.imshow(imagen_Rx)

# Visualizar el cambio entre las señales
fig, (ax2, ax3, ax4) = plt.subplots(nrows=3, sharex=True, figsize=(14, 7))

# La señal modulada por 16-QAM
ax2.plot(senal_Tx[0:600], color='g', lw=2)
ax2.set_ylabel('$s(t)$')

# La señal modulada al dejar el canal
ax3.plot(senal_Rx[0:600], color='b', lw=2)
ax3.set_ylabel('$s(t) + n(t)$')

# La señal demodulada
ax4.plot(senal_demodulada[0:600], color='m', lw=2)
ax4.set_ylabel('$b^{\'}(t)$')
ax4.set_xlabel('$t$ / milisegundos')
fig.tight_layout()
plt.show()

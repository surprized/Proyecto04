# Descripción de la solución

Para los graciosos que vendrán a GitHub para copiar mi tarea,
lean el archivo LICENSE.

En otras palabras, si me se "inspiran" en mi tarea, deben incluir
la licencia y mi nombre en su trabajo. _Si no lo hacen les tiro
mi EqUiPo dE aBoGaDos_.

## Introducción
Después de quebrarme la cabeza por 12 horas seguidas,
logré resolver los bugs que tenía mi código.

La "prueba de energía" funciona para demodulación, con
el cuidado de que para la segunda portadora la prueba
se invierte. Además se debe de agregar una prueba de amplitud, 
para la cual se eligió 2.5 como punto medio.

Al ajustar la SNR a 120 se obtiene cero errores de transmisión, pero
si se baja hasta ser negativa, la cifra de errores aumenta a ritmo
acelerado - esto tiene sentido pues al tratarse de modulación
por amplitud (además de fase), la señal pierde inmunidad al ruido.

La mayoría del código fue heredado del profesor.

## Modulador y demodulador
Se crea una expresión matemática que calcula la onda a emitir con base
en los cuatro bits del 16-QAM y las dos ondas portadoras (seno y coseno).

```python
Pulso = (-1)**(1+bits[i]) * 3**(1-bits[i+1]) * portadora1 + (-1)**(bits[i+2]) * 3**(1-bits[i+3]) * portadora2
```

El demodulador analiza la amplitud (valor máximo por conjunto de muestras)
y fase (prueba de energía), con el fin de extraer los cuatro bits por "pulso".

La creación de las funciones e investigación sobre el funcionamiento del 
16-QAM duró, sumadas, aproximadamente 13 horas.

Para la SNR, se obtiene una transmisión limpia:

![SNR = 120 \[dB\]](res/txrx.png)

Las señales asociadas son las siguientes:

![Señales transmitidas](res/signals.png)

## Densidad espectral de potencia
Directamente se utilizó el código proporcionado por el profesor. Se observa
un pico en los 5000 \[Hz\], lo cual es de esperar pues a ésta frecuencia es
que se transmite originalmente.

![Densidad de potencia](res/sxx.png)

## Estacionaridad y ergodicidad

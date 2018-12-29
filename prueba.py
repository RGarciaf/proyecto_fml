import matplotlib.image as mpimg
from Dataset import Dataset
import letritas
import numpy as np
np.set_printoptions(threshold=np.nan)


dataset = Dataset(seed=1)

imagen = mpimg.imread('imagenes/out-0032.png', True)

h, v = letritas.detectar_lineas(imagen)
'''
print ("h", h)
print ("v", v)
'''

# Se prueba a imprimir una sola imagen
imagenLetra = letritas.extrae_cuadradito(imagen, h[0], v[0], alto=200, ancho=150)
imagen_aux = dataset.todoBlancoNegro(imagenLetra)

dataset.mostrarImagen(imagen_aux)


# Se prueba a imprimir varias imagenes
imagenes = []
num_columnas = 10 # Numero de columnas a mostrar en la imagen

imagenLetra00 = letritas.extrae_cuadradito(imagen, h[0], v[0], alto=200, ancho=150)
imagenLetra01 = letritas.extrae_cuadradito(imagen, h[0], v[1], alto=200, ancho=150)
imagenLetra10 = letritas.extrae_cuadradito(imagen, h[1], v[0], alto=200, ancho=150)
imagenLetra11 = letritas.extrae_cuadradito(imagen, h[1], v[1], alto=200, ancho=150)

imagenes.append(dataset.todoBlancoNegro(imagenLetra00))
imagenes.append(dataset.todoBlancoNegro(imagenLetra01))
imagenes.append(dataset.todoBlancoNegro(imagenLetra10))
imagenes.append(dataset.todoBlancoNegro(imagenLetra11))

dataset.mostrarImagenes(imagenes, columnas=num_columnas)



# Se prueba a imprimir en varias imagenes trozos de una sola letra
num_columnas = 10 # Numero de columnas a mostrar en la imagen

x_ini, x_fin, y_ini, y_fin = dataset.quitaMargenes(h[0], v[0])

recortes_imagen = dataset.crearCuadraditos(x_ini=x_ini, x_fin=x_fin, y_ini=y_ini, y_fin=y_fin, n_pixeles_ancho=None, n_pixeles_alto=25)

imagenes = []

for recorte in recortes_imagen:
	imagen_recorte = imagen[recorte[0][0]:recorte[0][1], recorte[1][0]:recorte[1][1]]
	imagenes.append(dataset.todoBlancoNegro(imagen_recorte))

num_columnas = len(imagenes)

dataset.mostrarImagenes(imagenes, columnas=num_columnas)




# Se comprueba como algunas zonas de la imagen salen negras y por eso se subio
# el umbral de que fuese negro de 150 a 165 en la funcion datasettodoBlancoNegro()
# Cuidado no subir el umbral por encima del 180, los resultados son bastante horribles
imagenes = []
# Se prueba a imprimir una sola imagen
#imagenLetra = letritas.extrae_cuadradito(imagen, h[0], v[0], alto=200, ancho=150)
imagenes.append(imagen[646:846, 480:510])
imagenes.append(imagen[646:846, 400:550])
imagenes.append(imagen[646:846, 420:540])
imagenes.append(imagen[646:846, 440:530])
imagenes.append(imagen[646:846, 460:520])
imagenes.append(imagen[646:846, 480:510])
imagenes.append(imagen[646:846, 479:509])
imagenes.append(imagen[646:846, 479:508])
imagenes.append(imagen[646:846, 479:507])
imagenes.append(imagen[646:846, 479:506])
imagenes.append(imagen[646:846, 479:505])
imagenes.append(imagen[646:846, 479:504])

imagenes.append(dataset.todoBlancoNegro(imagen[646:846, 480:510]))
imagenes.append(dataset.todoBlancoNegro(imagen[646:846, 400:550]))
imagenes.append(dataset.todoBlancoNegro(imagen[646:846, 420:540]))
imagenes.append(dataset.todoBlancoNegro(imagen[646:846, 440:530]))
imagenes.append(dataset.todoBlancoNegro(imagen[646:846, 460:520]))
imagenes.append(dataset.todoBlancoNegro(imagen[646:846, 480:510]))
imagenes.append(dataset.todoBlancoNegro(imagen[646:846, 479:509]))
imagenes.append(dataset.todoBlancoNegro(imagen[646:846, 479:508]))
imagenes.append(dataset.todoBlancoNegro(imagen[646:846, 479:507]))
imagenes.append(dataset.todoBlancoNegro(imagen[646:846, 479:506]))
imagenes.append(dataset.todoBlancoNegro(imagen[646:846, 479:505]))
imagenes.append(dataset.todoBlancoNegro(imagen[646:846, 479:504]))

dataset.mostrarImagenes(imagenes, columnas=len(imagenes)/2)


import matplotlib.image as mpimg
from Dataset import Dataset
import letritas
import numpy as np
import math
np.set_printoptions(threshold=np.nan)



'''
####################################################################################################################
# Comienzo de la seccion 1 del  Main de prueba, en esta seccion
# se utilizaran la imagenes completas de las hojas con letras
####################################################################################################################
dataset = Dataset(seed=1)

imagen = mpimg.imread('imagenes/out-0032.png', True)

h, v = letritas.detectar_lineas(imagen)
'''
'''
print ("h", h)
print ("v", v)
'''
'''
# Se prueba a imprimir una sola imagen
imagenLetra = letritas.extrae_cuadradito(imagen, h[0], v[0], alto=200, ancho=150)
imagen_aux = dataset.todoBlancoNegro(imagenLetra)

dataset.mostrarImagen(imagen_aux)
'''
'''
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
'''

'''
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
'''


####################################################################################################################
# Comienzo de la seccion 2 del Main de prueba, en esta seccion se utilizaran la imagenes
# de letras de la carpeta out y se mostrara el funcionamientos de la funcion mostrarImagenes
####################################################################################################################
'''
n_pixeles_ancho = 100
n_pixeles_alto = 100

dataset = Dataset(seed=1)

imagen = mpimg.imread('out/l00000_A.png', True)
imagenBN = dataset.todoBlancoNegro(imagen)

l_cuadraditros = dataset.crearCuadraditos(0, imagenBN.shape[1], 0, imagenBN.shape[0], n_pixeles_ancho=n_pixeles_ancho, n_pixeles_alto=n_pixeles_alto)
dataset.cuadraditosRandom(imagenBN, l_cuadraditros, porcentajeAgrupacion=0.1, solo_blanco_negro=True, random=False)


imagenes = []
for cuadradito in l_cuadraditros:
    imagen_recorte = imagenBN[cuadradito[1][0]:cuadradito[1][1], cuadradito[0][0]:cuadradito[0][1]]
    imagenes.append(imagen_recorte)

dataset.mostrarImagenes(imagenes, columnas=math.ceil(imagenBN.shape[1]/n_pixeles_ancho))
'''
####################################################################################################################
# Comienzo de la seccion 3 del Main de prueba, en esta seccion se utilizaran la imagenes
# de letras de la carpeta out y se mostrara el funcionamientos de la funcion random
####################################################################################################################
'''
n_pixeles_ancho = 10 # No se recomienda utilizar numeros inferiores a 6 porque se distorsiona todo
n_pixeles_alto = 10 # No se recomienda utilizar numeros inferiores a 6 porque se distorsiona todo

dataset = Dataset(seed=1)

imagen = mpimg.imread('out/l00000_A.png', True)
imagenBN = dataset.todoBlancoNegro(imagen)

l_cuadraditros = dataset.crearCuadraditos(0, imagenBN.shape[1], 0, imagenBN.shape[0], n_pixeles_ancho=n_pixeles_ancho, n_pixeles_alto=n_pixeles_alto)
cuadraditos_random = dataset.cuadraditosRandom(imagenBN, 0, l_cuadraditros, porcentajeAgrupacion=0.001, solo_blanco_negro=True, random=False)

i_1 = 0
i_2 = imagenBN.shape[1]/n_pixeles_ancho
tam_cuadraditos_random = len(cuadraditos_random)
while i_2 < tam_cuadraditos_random:
    print (cuadraditos_random[round(i_1):round(i_2)])
    i_1 = i_2
    i_2 += imagenBN.shape[1]/n_pixeles_ancho
'''

####################################################################################################################
# Comienzo de la seccion 4 del Main de prueba, en esta seccion se utilizaran la imagenes
# de letras de la carpeta out y se creara el conjunto de datos correspondiente a random
####################################################################################################################
'''
n_pixeles_ancho = 15 # No se recomienda utilizar numeros inferiores a 6 porque se distorsiona todo
n_pixeles_alto = 15 # No se recomienda utilizar numeros inferiores a 6 porque se distorsiona todo

dataset = Dataset(seed=1)

letras = "ABCDEFGHIJ"

lista_imagenes = []
for i in range(0, 1320):
    numero = str(100000 + i)
    lista_imagenes.append("l"+ numero[1:] + "_" + letras[i%10])

l_atributos = []
l_clases = []
for nombre_imagen in lista_imagenes[:10]:
    imagen = mpimg.imread('out/' + nombre_imagen + '.png', True)
    imagenBN = dataset.todoBlancoNegro(imagen)

    l_cuadraditros = dataset.crearCuadraditos(0, imagenBN.shape[1], 0, imagenBN.shape[0], n_pixeles_ancho=n_pixeles_ancho, n_pixeles_alto=n_pixeles_alto)
    l_atributos.append(dataset.cuadraditosRandom(imagenBN, l_cuadraditros, porcentajeAgrupacion=0.001, solo_blanco_negro=True, random=False))
    l_clases.append(nombre_imagen[-1])

dataset.crearDataset(l_atributos, l_clases, ruta="ConjuntosDatos/", nombre="letritasRandom.data")
'''


####################################################################################################################
# Comienzo de la seccion 5 del Main de prueba, en esta seccion se utilizaran la imagenes
# de letras de la carpeta out y se probara la funcion recortar
####################################################################################################################

'''
dataset = Dataset(seed=1)

imagen = mpimg.imread('out/l00000_A.png', True)
imagen_recortada = dataset.recortar(imagen, porcentaje_umbral_recorte=.1)
dataset.mostrarImagen(imagen_recortada)
imagen_recortada = dataset.recortar(imagen, porcentaje_umbral_recorte=[0.05, 0.1, 0.08, 0.08])
dataset.mostrarImagen(imagen_recortada)

imagen = mpimg.imread('out/l00028_I.png', True)
imagen_recortada = dataset.recortar(imagen, porcentaje_umbral_recorte=.1)
dataset.mostrarImagen(imagen_recortada)
imagen_recortada = dataset.recortar(imagen, porcentaje_umbral_recorte=0.1, redimensionar=False)
dataset.mostrarImagen(imagen_recortada)
'''


####################################################################################################################
# Comienzo de la seccion 6 del Main de prueba, en esta seccion se utilizaran la imagenes
# de letras de la carpeta out y se probara la funcion patrones
####################################################################################################################

'''
dataset = Dataset(seed=1)

letras = "ABCDEFGHIJ"

lista_imagenes = []
for i in range(20, 29):
    numero = str(100000 + i)
    lista_imagenes.append(mpimg.imread('out/' + "l"+ numero[1:] + "_" + letras[i%10] + '.png', True))

for i, imagen in enumerate(lista_imagenes):
    imagen = dataset.todoBlancoNegro(imagen)
    l_patrones = dataset.patrones(imagen, i%len(letras), solo_blanco_negro=True).tolist()
    l_patrones.pop()

    for i in range(0, len(l_patrones), 7):
        print (l_patrones[i+0], l_patrones[i+1], l_patrones[i+2], l_patrones[i+3], l_patrones[i+4], l_patrones[i+5], l_patrones[i+6])
    
    print ()
    print ()
'''


import matplotlib.image as mpimg
import numpy as np
import math
import statistics

import letritas
from Dataset import Dataset
import EstrategiaParticionadoSL
import ClasificadorSL



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
n_pixeles_ancho = 20
n_pixeles_alto = 20

dataset = Dataset(seed=1)

imagen = mpimg.imread('out/l00000_A.png', True)
imagenBN = dataset.todoBlancoNegro(imagen)



l_cuadraditros = dataset.crearCuadraditos(0, imagenBN.shape[1], 0, imagenBN.shape[0], n_pixeles_ancho=n_pixeles_ancho, n_pixeles_alto=n_pixeles_alto)
dataset.cuadraditosRandom(imagenBN, 0, l_cuadraditros, porcentajeAgrupacion=0.1, solo_blanco_negro=False, random=False)


imagenes = []
tam_base = -1
for cuadradito in l_cuadraditros:
    if tam_base != -1:
        imagen_recorte = imagenBN[cuadradito[1][0]:cuadradito[1][1], cuadradito[0][0]:cuadradito[0][1]]
        if imagen_recorte.size >= 0.8*tam_base:
            imagenes.append(imagen_recorte)
    else:

        imagen_recorte = imagenBN[cuadradito[1][0]:cuadradito[1][1], cuadradito[0][0]:cuadradito[0][1]]
        imagenes.append(imagen_recorte)
        tam_base = imagen_recorte.size


dataset.mostrarImagenes(imagenes, math.floor(imagenBN.shape[1]/n_pixeles_ancho))
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

#l_cuadraditros = dataset.crearCuadraditos(0, imagenBN.shape[1], 0, imagenBN.shape[0], n_pixeles_ancho=n_pixeles_ancho, n_pixeles_alto=n_pixeles_alto)
#cuadraditos_random = dataset.cuadraditosRandom(imagenBN, 0, l_cuadraditros, porcentajeAgrupacion=0.001, solo_blanco_negro=True, random=False)

letra = np.array([1,2])

dataset.mostrarImagen(np.reshape(np.array(dataset.cuadraditos(imagen, 0, tamCuadrado=12, solo_blanco_negro=True)[:-1], dtype=np.int), (17, 12)))
letra[1] = np.reshape(np.array(dataset.cuadraditos(imagen, 0, tamCuadrado=12, solo_blanco_negro=True)[:-1], dtype=np.int), (12, 17))

dataset.mostrarImagenes(letra, columnas=2)
'''
'''
i_1 = 0
i_2 = 12
while i_2 < 12*16:
    print (letra2[round(i_1):round(i_2)],letra[round(i_1):round(i_2)])
    i_1 = i_2
    i_2 += 12
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
imagen_recortada = dataset.recortar(imagen, porcentaje_umbral_recorte=[0.22, 0.22, 0.08, 0.08], redimensionar=False)
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

for i, imagen in enumerate(lista_imagenes[0:1]):
    #imagen = dataset.todoBlancoNegro(imagen)
    imagen = dataset.recortar(imagen, porcentaje_umbral_recorte=[0.1, 0.05, 0.05, 0.15], redimensionar=True)
    l_patrones = dataset.patrones(imagen, i%len(letras), solo_blanco_negro=False).tolist()
    l_patrones.pop()
'''

'''    
    for i in range(0, len(l_patrones), 7):
        print (l_patrones[i+0], l_patrones[i+1], l_patrones[i+2], l_patrones[i+3], l_patrones[i+4], l_patrones[i+5], l_patrones[i+6])
    
    print ()
    print ()
'''

'''
print (l_patrones)
imagenes = []
for cuadradito in l_patrones:
    imagenes.append(imagen[cuadradito[1][0]:cuadradito[1][1], cuadradito[0][0]:cuadradito[0][1]])

dataset.mostrarImagenes(imagenes, 7)#math.floor(imagen.shape[1]/n_pixeles_ancho))
'''


seed=1


# Dataset
nombres_imagenes = sorted(letritas.parametros_por_defecto("out/")[0])
celdas = []
for nombre_imagen in nombres_imagenes:
    celdas.append(mpimg.imread(nombre_imagen, True))



tipo_atributos=["cuadraditos", "filas", "columnas"]
tamano=33
random=[False, True]



# Val cruzada
val_cruzada = EstrategiaParticionadoSL.ValidacionCruzadaSL(numeroParticiones=5)

with open("Datos_random_patrones/prueba.txt", 'w') as fichero:

    '''
    for i_tipo, tipo_atributo in enumerate(tipo_atributos):
        for i_pixeles in range(num_atributos_n_pixeles):
            for porcentajeAgrupacion in porcentajeAgrupaciones[i_tipo][i_pixeles]:
                for hacer_recorte in hacer_recortes:
    '''
    for tipo_atributo in tipo_atributos:
        for _random in random:
            fichero.write("tipo_atributo\n" + str(tipo_atributo))
            fichero.write("tamano\n" + str(tamano))
            fichero.write("_random\n" + str(_random))


            dataset = Dataset(seed=seed)
            dataset.procesarDatos(celdas)

            dataset.procesarCuadraditos(tipo_atributo, tamano=tamano, random=_random)

            # Clasificadores
            clasificadorSL_KNN_uniform = ClasificadorSL.ClasificadorKNN_SL()
            clasificadorSL_KNN_distance = ClasificadorSL.ClasificadorKNN_SL("distance")

            clasificadorSL_NB = ClasificadorSL.ClasificadorNB_SL()
            clasificadorSL_RegLog = ClasificadorSL.ClasificadorRegLog_SL(num_epocas=10)
            clasificadorSL_ArbolDecision = ClasificadorSL.ClasificadorArbolDecision_SL()
            clasificadorSL_RandomForest = ClasificadorSL.ClasificadorRandomForest_SL()

            # Cuadraditos
            errores_particion_NB = clasificadorSL_NB.validacion(val_cruzada, dataset, clasificadorSL_NB, seed=seed)
            errores_particion_KNN = clasificadorSL_KNN_uniform.validacion(val_cruzada, dataset, clasificadorSL_KNN_uniform, seed=seed)
            errores_particion_KNN_2 = clasificadorSL_KNN_distance.validacion(val_cruzada, dataset, clasificadorSL_KNN_distance, seed=seed)
            errores_particion_RL = clasificadorSL_RegLog.validacion(val_cruzada, dataset, clasificadorSL_RegLog, seed=seed)
            errores_particion_Tree = clasificadorSL_ArbolDecision.validacion(val_cruzada, dataset, clasificadorSL_ArbolDecision, seed=seed)
            errores_particion_RF = clasificadorSL_RandomForest.validacion(val_cruzada, dataset, clasificadorSL_RandomForest, seed=seed)


            fichero.write("GaussianNB\t\t\tKNN_uniform\t\t\tKNN_distance\t\tRegresionLog\t\tTreeDecision\t\tRandomForest\t\t\n" +
                          str(round(statistics.mean(errores_particion_NB), 4)) + "+-" + str(round(statistics.stdev(errores_particion_NB), 4)) + "\t\t" +
                          str(round(statistics.mean(errores_particion_KNN), 4)) + "+-" + str(round(statistics.stdev(errores_particion_KNN), 4)) + "\t\t" +
                          str(round(statistics.mean(errores_particion_KNN_2), 4)) + "+-" + str(round(statistics.stdev(errores_particion_KNN_2), 4)) + "\t\t" +
                          str(round(statistics.mean(errores_particion_RL), 4)) + "+-" + str(round(statistics.stdev(errores_particion_RL), 4)) + "\t\t" +
                          str(round(statistics.mean(errores_particion_Tree), 4)) + "+-" + str(round(statistics.stdev(errores_particion_Tree), 4)) + "\t\t" +
                          str(round(statistics.mean(errores_particion_RF), 4)) + "+-" + str(round(statistics.stdev(errores_particion_RF), 4))
                              + "\n-----------------------------------------------------------------------------------------------------------------------\n\n")

                    #raise ValueError("parar")

fichero.close()




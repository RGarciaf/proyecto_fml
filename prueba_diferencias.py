import matplotlib.image as mpimg    

from Dataset import Dataset
import EstrategiaParticionadoSL
import ClasificadorSL
import letritas
import statistics
import pprint
import numpy as np
np.set_printoptions(threshold=np.nan)


seed=1

# Generar el dataset y la estrategia de particionado
#celdas = letritas.run(letritas.parametros_por_defecto)

nombres_imagenes = sorted(letritas.parametros_por_defecto("out/")[0])
celdas = []
for nombre_imagen in nombres_imagenes:
    celdas.append(mpimg.imread(nombre_imagen, True))

dataset = Dataset(seed=seed)
dataset.procesarDatos(celdas)
dataset.diferenciaPixel(arreglo="abs")

imagenes = []
for imag in dataset.datos:
	imagenes.append(np.reshape(imag[:-2], (206,150)))

dataset.mostrarImagenes(imagenes, 5)

'''
val_cruzada = EstrategiaParticionadoSL.ValidacionCruzadaSL(numeroParticiones=5)


# Prueba clasificador NB
clasificadorSL_NB = ClasificadorSL.ClasificadorNB_SL()


errores_particion = clasificadorSL_NB.validacion(val_cruzada, dataset, clasificadorSL_NB, seed=seed)
aciertos_particion = [(1 - elem) for elem in errores_particion]

if clasificadorSL_NB.Multinomial_flag == True:
    print("MultinomialNB")
else:
    print("GaussianNB")
#print("Errores particion: " + str(errores_particion))
#print("Media: " + str(statistics.mean(errores_particion)))
#print("Desv tipica: " + str(statistics.stdev(errores_particion)))

print("Aciertos particion: " + str(aciertos_particion))
print("Media: " + str(statistics.mean(aciertos_particion)))
print("Desv tipica: " + str(statistics.stdev(aciertos_particion)))
print()


# Prueba clasificador KNN
clasificadorSL_KNN = ClasificadorSL.ClasificadorKNN_SL("distance", 3)

errores_particion = clasificadorSL_KNN.validacion(val_cruzada, dataset, clasificadorSL_KNN, seed=seed)
aciertos_particion = [(1 - elem) for elem in errores_particion]

print("KNN")
#print("Errores particion: " + str(errores_particion))
#print("Media: " + str(statistics.mean(errores_particion)))
#print("Desv tipica: " + str(statistics.stdev(errores_particion)))

print("Aciertos particion: " + str(aciertos_particion))
print("Media: " + str(statistics.mean(aciertos_particion)))
print("Desv tipica: " + str(statistics.stdev(aciertos_particion)))
print()

# Prueba clasificador Regresion Logistica
clasificadorSL_RegLog = ClasificadorSL.ClasificadorRegLog_SL(num_epocas=10)

errores_particion = clasificadorSL_RegLog.validacion(val_cruzada, dataset, clasificadorSL_RegLog, seed=seed)
aciertos_particion = [(1 - elem) for elem in errores_particion]

print("Regresion Logistica")
#print("Errores particion: " + str(errores_particion))
#print("Media: " + str(statistics.mean(errores_particion)))
#print("Desv tipica: " + str(statistics.stdev(errores_particion)))

print("Aciertos particion: " + str(aciertos_particion))
print("Media: " + str(statistics.mean(aciertos_particion)))
print("Desv tipica: " + str(statistics.stdev(aciertos_particion)))
print()


# Prueba clasificador Arbol de Decision
clasificadorSL_ArbolDecision = ClasificadorSL.ClasificadorArbolDecision_SL()

errores_particion = clasificadorSL_ArbolDecision.validacion(val_cruzada, dataset, clasificadorSL_ArbolDecision, seed=seed)
aciertos_particion = [(1 - elem) for elem in errores_particion]

print("Arbol de Decision")
#print("Errores particion: " + str(errores_particion))
#print("Media: " + str(statistics.mean(errores_particion)))
#print("Desv tipica: " + str(statistics.stdev(errores_particion)))

print("Aciertos particion: " + str(aciertos_particion))
print("Media: " + str(statistics.mean(aciertos_particion)))
print("Desv tipica: " + str(statistics.stdev(aciertos_particion)))
print()


# Prueba clasificador Random Forest
clasificadorSL_RandomForest = ClasificadorSL.ClasificadorRandomForest_SL()

errores_particion = clasificadorSL_RandomForest.validacion(val_cruzada, dataset, clasificadorSL_RandomForest, seed=seed)
aciertos_particion = [(1 - elem) for elem in errores_particion]

print("Random Forest")
#print("Errores particion: " + str(errores_particion))
#print("Media: " + str(statistics.mean(errores_particion)))
#print("Desv tipica: " + str(statistics.stdev(errores_particion)))

print("Aciertos particion: " + str(aciertos_particion))
print("Media: " + str(statistics.mean(aciertos_particion)))
print("Desv tipica: " + str(statistics.stdev(aciertos_particion)))
'''

from Dataset import Dataset
import EstrategiaParticionadoSL
import ClasificadorSL
import letritas
import statistics
import numpy as np
np.set_printoptions(threshold=np.nan)

# Generar el dataset y la estrategia de particionado
celdas = letritas.run(letritas.parametros_por_defecto)


dataset = Dataset(seed=1)
dataset.procesarDatos(celdas)
dataset.procesarCuadraditos("cuadraditos", 10)

'''
tipo_procesado="cuadraditos"
n_pixeles_ancho=5
n_pixeles_alto=5
porcentajeAgrupacion=0.008
solo_blanco_negro=True
random=True
print ("tipo_procesado\t\t", tipo_procesado)
print ("n_pixeles_ancho\t\t", n_pixeles_ancho)
print ("n_pixeles_alto\t\t", n_pixeles_alto)
print ("porcentajeAgrupacion\t", porcentajeAgrupacion)
print ("solo_blanco_negro\t", solo_blanco_negro)
print ("random\t\t\t", random + "\n\n")

dataset.procesarCuadraditos(tipo_procesado, tamano=10,
                            n_pixeles_ancho=n_pixeles_ancho, n_pixeles_alto=n_pixeles_alto,
                            porcentajeAgrupacion=porcentajeAgrupacion, solo_blanco_negro=solo_blanco_negro, random=random)
'''

val_cruzada = EstrategiaParticionadoSL.ValidacionCruzadaSL(numeroParticiones=5)


# Prueba clasificador NB
clasificadorSL_NB = ClasificadorSL.ClasificadorNB_SL()


errores_particion = clasificadorSL_NB.validacion(val_cruzada, dataset, clasificadorSL_NB)
aciertos_particion = [(1 - elem) for elem in errores_particion]

if clasificadorSL_NB.Multinomial_flag == True:
    print("MultinomialNB")
else:
    print("GaussianNB")
#print("Errores particion: " + str(errores_particion))
#print("Media: " + str(statistics.mean(errores_particion)))

print("Aciertos particion: " + str(aciertos_particion))
print("Media: " + str(statistics.mean(aciertos_particion)))
print()


# Prueba clasificador KNN
clasificadorSL_KNN = ClasificadorSL.ClasificadorKNN_SL("distance", 3)

errores_particion = clasificadorSL_KNN.validacion(val_cruzada, dataset, clasificadorSL_KNN)
aciertos_particion = [(1 - elem) for elem in errores_particion]

print("KNN")
#print("Errores particion: " + str(errores_particion))
#print("Media: " + str(statistics.mean(errores_particion)))

print("Aciertos particion: " + str(aciertos_particion))
print("Media: " + str(statistics.mean(aciertos_particion)))
print()

# Prueba clasificador Regresion Logistica
clasificadorSL_RegLog = ClasificadorSL.ClasificadorRegLog_SL(num_epocas=10)

errores_particion = clasificadorSL_RegLog.validacion(val_cruzada, dataset, clasificadorSL_RegLog)
aciertos_particion = [(1 - elem) for elem in errores_particion]

print("Regresion Logistica")
#print("Errores particion: " + str(errores_particion))
#print("Media: " + str(statistics.mean(errores_particion)))

print("Aciertos particion: " + str(aciertos_particion))
print("Media: " + str(statistics.mean(aciertos_particion)))
print()


# Prueba clasificador Arbol de Decision
clasificadorSL_ArbolDecision = ClasificadorSL.ClasificadorArbolDecision_SL()

errores_particion = clasificadorSL_ArbolDecision.validacion(val_cruzada, dataset, clasificadorSL_ArbolDecision)
aciertos_particion = [(1 - elem) for elem in errores_particion]

print("Arbol de Decision")
#print("Errores particion: " + str(errores_particion))
#print("Media: " + str(statistics.mean(errores_particion)))

print("Aciertos particion: " + str(aciertos_particion))
print("Media: " + str(statistics.mean(aciertos_particion)))
print()


# Prueba clasificador Random Forest
clasificadorSL_RandomForest = ClasificadorSL.ClasificadorRandomForest_SL()

errores_particion = clasificadorSL_RandomForest.validacion(val_cruzada, dataset, clasificadorSL_RandomForest)
aciertos_particion = [(1 - elem) for elem in errores_particion]

print("Random Forest")
#print("Errores particion: " + str(errores_particion))
#print("Media: " + str(statistics.mean(errores_particion)))

print("Aciertos particion: " + str(aciertos_particion))
print("Media: " + str(statistics.mean(aciertos_particion)))






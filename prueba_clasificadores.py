import matplotlib.image as mpimg    

from Dataset import Dataset
import EstrategiaParticionadoSL
import ClasificadorSL
import letritas
import statistics
import numpy as np
np.set_printoptions(threshold=np.nan)


seed=1

# Generar el dataset y la estrategia de particionado
celdas = letritas.run(letritas.parametros_por_defecto)

dataset = Dataset(seed=seed)
dataset.procesarDatos(celdas)
# dataset.diferenciaPixel()
dataset.procesarCuadraditos("cuadraditos", 10)

# print(dataset.datos)
#dataset.procesarCuadraditos("patrones", 10, solo_blanco_negro=True, hacer_recorte=True)

'''
nombres_imagenes = sorted(letritas.parametros_por_defecto("out/")[0])
celdas = []
for nombre_imagen in nombres_imagenes:
    celdas.append(mpimg.imread(nombre_imagen, True))


tipo_atributo="random"
n_pixeles_ancho=10
n_pixeles_alto=n_pixeles_ancho
porcentajeAgrupacion=0.0
solo_blanco_negro=True
random=False
hacer_recorte=False
print ("tipo_atributo\t\t", tipo_atributo)
print ("n_pixeles_ancho\t\t", n_pixeles_ancho)
print ("n_pixeles_alto\t\t", n_pixeles_alto)
print ("porcentajeAgrupacion\t", porcentajeAgrupacion)
print ("solo_blanco_negro\t", solo_blanco_negro)
print ("random\t\t\t", str(random))
print ("hacer_recorte\t\t", str(hacer_recorte) + "\n\n")

dataset = Dataset(seed=seed)
dataset.procesarDatos(celdas)
dataset.procesarCuadraditos(tipo_atributo, tamano=10,
                            n_pixeles_ancho=n_pixeles_ancho, n_pixeles_alto=n_pixeles_alto,
                            porcentajeAgrupacion=porcentajeAgrupacion, solo_blanco_negro=solo_blanco_negro,
                            random=random, hacer_recorte=hacer_recorte)
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


import matplotlib.image as mpimg

from Dataset import Dataset
import EstrategiaParticionadoSL
import ClasificadorSL
import letritas
import statistics
import numpy as np
np.set_printoptions(threshold=np.nan)


seed=1
tipo_atributo="columnas"
tamano=20
#n_pixeles_ancho=9
#n_pixeles_alto=n_pixeles_ancho
#porcentajeAgrupacion=0.1
hacer_recorte=False
solo_blanco_negro=True
#random=False


# Dataset
nombres_imagenes = sorted(letritas.parametros_por_defecto("out/")[0])
celdas = []
for nombre_imagen in nombres_imagenes:
    celdas.append(mpimg.imread(nombre_imagen, True))

dataset = Dataset(seed=seed)
dataset.procesarDatos(celdas)

# Val cruzada
val_cruzada = EstrategiaParticionadoSL.ValidacionCruzadaSL(numeroParticiones=5)

# --------------------------
# CUADRADITOS
# --------------------------
# Pruebas con tamCuadrado=1, 10, 20
dataset.procesarCuadraditos("cuadraditos", tamano=tamano,  solo_blanco_negro=solo_blanco_negro, hacer_recorte=hacer_recorte)

# Clasificadores
clasificadorSL_KNN_uniform = ClasificadorSL.ClasificadorKNN_SL()
clasificadorSL_KNN_distance = ClasificadorSL.ClasificadorKNN_SL("distance")

clasificadorSL_NB = ClasificadorSL.ClasificadorNB_SL()
clasificadorSL_RegLog = ClasificadorSL.ClasificadorRegLog_SL(num_epocas=10)
clasificadorSL_ArbolDecision = ClasificadorSL.ClasificadorArbolDecision_SL()
clasificadorSL_RandomForest = ClasificadorSL.ClasificadorRandomForest_SL()

# Cuadraditos
errores_particion_KNN = clasificadorSL_KNN_uniform.validacion(val_cruzada, dataset, clasificadorSL_KNN_uniform, seed=seed)
errores_particion_KNN_2 = clasificadorSL_KNN_distance.validacion(val_cruzada, dataset, clasificadorSL_KNN_distance, seed=seed)
errores_particion_RL = clasificadorSL_RegLog.validacion(val_cruzada, dataset, clasificadorSL_RegLog, seed=seed)
errores_particion_Tree = clasificadorSL_ArbolDecision.validacion(val_cruzada, dataset, clasificadorSL_ArbolDecision, seed=seed)
errores_particion_RF = clasificadorSL_RandomForest.validacion(val_cruzada, dataset, clasificadorSL_RandomForest, seed=seed)


print(round(statistics.mean(errores_particion_KNN), 4), "+-", round(statistics.stdev(errores_particion_KNN), 4), "\t\t",
      round(statistics.mean(errores_particion_KNN_2), 4), "+-", round(statistics.stdev(errores_particion_KNN_2), 4), "\t\t",
      round(statistics.mean(errores_particion_RL), 4), "+-", round(statistics.stdev(errores_particion_RL), 4), "\t\t",
      round(statistics.mean(errores_particion_Tree), 4), "+-", round(statistics.stdev(errores_particion_Tree), 4), "\t\t",
      round(statistics.mean(errores_particion_RF), 4), "+-", round(statistics.stdev(errores_particion_RF), 4))
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
celdas = letritas.run(letritas.parametros_por_defecto)

dataset = Dataset(seed=seed)
dataset.procesarDatos(celdas)
dataset.diferenciaPixel()
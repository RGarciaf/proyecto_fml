from abc import ABCMeta, abstractmethod
from sklearn.model_selection import KFold


class Particion():

	# Constructor de la clase
	def __init__(self):
		self.indicesTrain = []
		self.indicesTest = []

	# Imprime el objeto Particion en forma de user-friendly string
	def __repr__(self):
		return "indicesTrain: " + str(self.indicesTrain) + " indicesTest: " + str(self.indicesTest) + "\n"

#####################################################################################################

class EstrategiaParticionado(object):
	# Clase abstracta
	__metaclass__ = ABCMeta

	# Constructor de la clase
	def __init__(self, nombreEstrategia, numeroParticiones):
		self.nombreEstrategia = nombreEstrategia
		self.numeroParticiones = numeroParticiones
		self.listaParticiones = []

	@abstractmethod
	# Crea una lista de particiones a partir del dataset
	def creaParticiones(self, datos, seed=None):
		pass

#####################################################################################################

# Validacion K-Folds
class ValidacionCruzadaSL(EstrategiaParticionado):

	# Constructor de la clase
	def __init__(self, nombreEstrategia="Validacion cruzada Sklearn", numeroParticiones=10):
		# Comprueba que el numero de particiones no sea inferior a 2
		if numeroParticiones < 2:
			raise ValueError("El numero de particiones en Validacion Cruzada no puede ser inferior a 2")

		# Llama al constructor de la clase padre abstracta
		super().__init__(nombreEstrategia, numeroParticiones)

	# Crea particiones segun el metodo de validacion cruzada.
	# El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
	def creaParticiones(self, datos, seed=None):

		val_cruzada = KFold(shuffle=True, n_splits=self.numeroParticiones)

		# Divide el dataset en K folds. Por cada uno, crea una particion, obtiene
		# las listas de indices de train y test, y las asocia a dicha particion.
		for indices_train, indices_test in val_cruzada.split(datos):
			# Crea la nueva particion
			particion = Particion()
			particion.indicesTrain = indices_train.tolist()
			particion.indicesTest = indices_test.tolist()

			# Annade la particion a la lista de particiones
			self.listaParticiones.append(particion)

		return

from abc import ABCMeta, abstractmethod
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class ClasificadorSL(object):

    # Clase abstracta
    __metaclass__ = ABCMeta

    # Entrena el modelo con los datos de entrenamiento pasados como argumento
    def entrenamiento(self, atributos_train, clases_train):
        self.modelo.fit(atributos_train, clases_train)

    # Predice las clases en funcion de los atributos de test
    def clasifica(self, atributos_test):
        return self.modelo.predict(atributos_test)

    # Obtiene el porcentaje de error del clasificador
    def error(self, clases_reales, predicciones):
        # accuracy_score compara las predicciones con las clases_reales y se calcula el error
        return round((1 - accuracy_score(clases_reales, predicciones)), 5)

    # Obtiene la matriz de confusion correspondiente
    def matrizConfusion(self, clases_reales, predicciones):
        # confusion_matrix compara las predicciones con las clases_reales y se calcula la matriz de confusion
        return confusion_matrix(clases_reales, predicciones)

    # Crea las particiones, entrena al clasificador y obtiene los porcentajes de error por cada particion
    @abstractmethod
    def validacion(self, particionadoSL, dataset, clasificadorSL, seed=None):
        pass

########################################################################################################################

# Naive Bayes
class ClasificadorNB_SL(ClasificadorSL):

    # Constructor de la clase
    def __init__(self, correccion_Laplace=1.0):
        self.Multinomial_flag = True
        self.correccion_Laplace = correccion_Laplace
        self.modelo = None

        self.matriz_confusion = None


    # Valida el modelo GaussianNB usando una estrategia de particionado determinada (tambien implementada con Scikit-learn)
    def validacion(self, particionadoSL, dataset, clasificadorSL, seed=None):

        # Se inicializan las listas de errores de las particiones
        errores_particion_MultinomialNB = []
        errores_particion_GaussianNB = []

        m_conf_particion_MultinomialNB = None
        m_conf_particion_GaussianNB = None

        # Crea las particiones de train y test
        particionadoSL.listaParticiones = []  # Reset de la lista por si se quiere ejecutar el codigo mas de una vez
        particionadoSL.creaParticiones(dataset.datos, seed)

        for particion in particionadoSL.listaParticiones:
            # Extrae los conjuntos de train y test
            datostrain = dataset.datos[particion.indicesTrain]
            datostest = dataset.datos[particion.indicesTest]

            # Codifica los conjuntos de train y test (X -> atributos, Y -> clases)
            X_train = datostrain[:, : -1]
            Y_train = datostrain[:, -1]

            X_test = datostest[:, : -1]
            Y_test = datostest[:, -1]

            # Se entrena el modelo MultinomialNB y se clasifica
            clasificadorSL.modelo = MultinomialNB(alpha=clasificadorSL.correccion_Laplace)
            clasificadorSL.entrenamiento(X_train, Y_train)

            predicciones = clasificadorSL.clasifica(X_test)

            # Se obtiene el error en las clasificaciones de MultinomialNB
            errores_particion_MultinomialNB.append(clasificadorSL.error(Y_test, predicciones))
            if m_conf_particion_MultinomialNB is None:
                m_conf_particion_MultinomialNB = clasificadorSL.matrizConfusion(Y_test, predicciones)
            else:
                m_conf_particion_MultinomialNB += clasificadorSL.matrizConfusion(Y_test, predicciones)

            # Se entrena el modelo GaussianNB y se clasifica
            clasificadorSL.modelo = GaussianNB()
            clasificadorSL.entrenamiento(X_train, Y_train)

            predicciones = clasificadorSL.clasifica(X_test)

            # Se obtiene el error en las clasificaciones de GaussianNB
            errores_particion_GaussianNB.append(clasificadorSL.error(Y_test, predicciones))
            if m_conf_particion_GaussianNB is None:
                m_conf_particion_GaussianNB = clasificadorSL.matrizConfusion(Y_test, predicciones)
            else:
                m_conf_particion_GaussianNB += clasificadorSL.matrizConfusion(Y_test, predicciones)

        # Comprueba que modelo ha tenido menos error
        if sum(errores_particion_MultinomialNB) < sum(errores_particion_GaussianNB):
            # Si se ha obtenido menos error con MultinomialNB se devuelve el
            # error de MultinomialNB (el flag Multinomial_flag ya esta a True)

            self.matriz_confusion = m_conf_particion_MultinomialNB/(len(particionadoSL.listaParticiones))
            return errores_particion_MultinomialNB

        else:
            # Si se ha obtenido menos error con GaussianNB se devuelve el
            # error de GaussianNB y se pone a False el flag Multinomial_flag
            clasificadorSL.Multinomial_flag = False

            self.matriz_confusion = m_conf_particion_GaussianNB/(len(particionadoSL.listaParticiones))
            return errores_particion_GaussianNB

########################################################################################################################

# Vecinos Proximos
class ClasificadorKNN_SL(ClasificadorSL):

    # Variable de clase
    TipoDePeso = ("uniform", "distance")

    # Constructor de la clase
    def __init__(self, tipo_peso="uniform", K=1):
        # Comprueba si el tipo de peso es "uniform" o "distance"
        if tipo_peso not in ClasificadorKNN_SL.TipoDePeso:
            raise ValueError("Error: Tipo de peso incorrecto: ", str(tipo_peso))

        self.tipo_peso = tipo_peso
        self.K = K
        self.modelo = KNeighborsClassifier(n_neighbors=self.K, weights=self.tipo_peso)

        self.matriz_confusion = None


    # Valida el modelo KNN usando una estrategia de particionado determinada (tambien implementada con Scikit-learn)
    def validacion(self, particionadoSL, dataset, clasificadorSL, seed=None):

        # Se inicializan las listas de errores de las particiones
        errores_particion = []

        # Crea las particiones de train y test
        particionadoSL.listaParticiones = []  # Reset de la lista por si se quiere ejecutar el codigo mas de una vez
        particionadoSL.creaParticiones(dataset.datos, seed)

        for particion in particionadoSL.listaParticiones:
            # Extrae los conjuntos de train y test
            datostrain = dataset.datos[particion.indicesTrain]
            datostest = dataset.datos[particion.indicesTest]

            # Codifica los conjuntos de train y test (X -> atributos, Y -> clases)
            X_train = datostrain[:, : -1]
            Y_train = datostrain[:, -1]

            X_test = datostest[:, : -1]
            Y_test = datostest[:, -1]

            # Se entrena y clasifica
            clasificadorSL.entrenamiento(X_train, Y_train)
            predicciones = clasificadorSL.clasifica(X_test)

            # Se obtiene el error en las clasificaciones
            errores_particion.append(clasificadorSL.error(Y_test, predicciones))
            if self.matriz_confusion is None:
                self.matriz_confusion = clasificadorSL.matrizConfusion(Y_test, predicciones)
            else:
                self.matriz_confusion += clasificadorSL.matrizConfusion(Y_test, predicciones)


        self.matriz_confusion = self.matriz_confusion/(len(particionadoSL.listaParticiones))
        # Se devuelve la lista de errores en las clasificaciones
        return errores_particion

########################################################################################################################

# Regresion Logistica
class ClasificadorRegLog_SL(ClasificadorSL):

    # Constructor de la clase
    def __init__(self, num_epocas=1, cte_aprendizaje=1.0):
        self.num_epocas = num_epocas
        self.cte_aprendizaje = cte_aprendizaje
        self.modelo = SGDClassifier(loss='log', alpha=self.cte_aprendizaje, \
                                        learning_rate='optimal', max_iter=self.num_epocas)

        self.matriz_confusion = None


    # Valida el modelo SGDClassifier usando una estrategia de particionado determinada (tambien implementada con Scikit-learn)
    def validacion(self, particionadoSL, dataset, clasificadorSL, seed=None):
        # Se inicializan las listas de errores de las particiones
        errores_particion = []

        # Crea las particiones de train y test
        particionadoSL.listaParticiones = []  # Reset de la lista por si se quiere ejecutar el codigo mas de una vez
        particionadoSL.creaParticiones(dataset.datos, seed)

        for particion in particionadoSL.listaParticiones:
            # Extrae los conjuntos de train y test
            datostrain = dataset.datos[particion.indicesTrain]
            datostest = dataset.datos[particion.indicesTest]

            # Codifica los conjuntos de train y test (X -> atributos, Y -> clases)
            X_train = datostrain[:, : -1]
            Y_train = datostrain[:, -1]

            X_test = datostest[:, : -1]
            Y_test = datostest[:, -1]

            # Se entrena y clasifica
            clasificadorSL.entrenamiento(X_train, Y_train)
            predicciones = clasificadorSL.clasifica(X_test)

            # Se obtiene el error en las clasificaciones
            errores_particion.append(clasificadorSL.error(Y_test, predicciones))
            if self.matriz_confusion is None:
                self.matriz_confusion = clasificadorSL.matrizConfusion(Y_test, predicciones)
            else:
                self.matriz_confusion += clasificadorSL.matrizConfusion(Y_test, predicciones)


        self.matriz_confusion = self.matriz_confusion/(len(particionadoSL.listaParticiones))
        # Se devuelve la lista de errores en las clasificaciones
        return errores_particion

########################################################################################################################

# Arbol de Decision
class ClasificadorArbolDecision_SL(ClasificadorSL):

    # Constructor de la clase
    def __init__(self, profundidad_maxima=None, min_ejemplos_split=2, min_ejemplos_en_hoja=1):
        self.profundidad_maxima = profundidad_maxima
        self.min_ejemplos_split = min_ejemplos_split
        self.min_ejemplos_en_hoja = min_ejemplos_en_hoja
        self.modelo = DecisionTreeClassifier(max_depth=self.profundidad_maxima, min_samples_split=self.min_ejemplos_split, \
                                             min_samples_leaf=self.min_ejemplos_en_hoja)

        self.matriz_confusion = None


    # Valida el modelo DecisionTreeClassifier usando una estrategia de particionado determinada (tambien implementada con Scikit-learn)
    def validacion(self, particionadoSL, dataset, clasificadorSL, seed=None):
        # Se inicializan las listas de errores de las particiones
        errores_particion = []

        # Crea las particiones de train y test
        particionadoSL.listaParticiones = []  # Reset de la lista por si se quiere ejecutar el codigo mas de una vez
        particionadoSL.creaParticiones(dataset.datos, seed)

        for particion in particionadoSL.listaParticiones:
            # Extrae los conjuntos de train y test
            datostrain = dataset.datos[particion.indicesTrain]
            datostest = dataset.datos[particion.indicesTest]

            # Codifica los conjuntos de train y test (X -> atributos, Y -> clases)
            X_train = datostrain[:, : -1]
            Y_train = datostrain[:, -1]

            X_test = datostest[:, : -1]
            Y_test = datostest[:, -1]

            # Se entrena y clasifica
            clasificadorSL.entrenamiento(X_train, Y_train)
            predicciones = clasificadorSL.clasifica(X_test)

            # Se obtiene el error en las clasificaciones
            errores_particion.append(clasificadorSL.error(Y_test, predicciones))
            if self.matriz_confusion is None:
                self.matriz_confusion = clasificadorSL.matrizConfusion(Y_test, predicciones)
            else:
                self.matriz_confusion += clasificadorSL.matrizConfusion(Y_test, predicciones)


        self.matriz_confusion = self.matriz_confusion/(len(particionadoSL.listaParticiones))
        # Se devuelve la lista de errores en las clasificaciones
        return errores_particion

########################################################################################################################

# Random Forest
class ClasificadorRandomForest_SL(ClasificadorSL):

    # Constructor de la clase
    def __init__(self, profundidad_maxima=None, num_estimadores=100, min_ejemplos_split=2, min_ejemplos_en_hoja=1):
        self.profundidad_maxima = profundidad_maxima
        self.num_estimadores = num_estimadores
        self.min_ejemplos_split = min_ejemplos_split
        self.min_ejemplos_en_hoja = min_ejemplos_en_hoja
        self.modelo = RandomForestClassifier(n_estimators=self.num_estimadores, max_depth=self.profundidad_maxima, \
                                             min_samples_split=self.min_ejemplos_split, min_samples_leaf=self.min_ejemplos_en_hoja)

        self.matriz_confusion = None


    # Valida el modelo RandomForestClassifier usando una estrategia de particionado determinada (tambien implementada con Scikit-learn)
    def validacion(self, particionadoSL, dataset, clasificadorSL, seed=None):
        # Se inicializan las listas de errores de las particiones
        errores_particion = []

        # Crea las particiones de train y test
        particionadoSL.listaParticiones = []  # Reset de la lista por si se quiere ejecutar el codigo mas de una vez
        particionadoSL.creaParticiones(dataset.datos, seed)

        for particion in particionadoSL.listaParticiones:
            # Extrae los conjuntos de train y test
            datostrain = dataset.datos[particion.indicesTrain]
            datostest = dataset.datos[particion.indicesTest]

            # Codifica los conjuntos de train y test (X -> atributos, Y -> clases)
            X_train = datostrain[:, : -1]
            Y_train = datostrain[:, -1]

            X_test = datostest[:, : -1]
            Y_test = datostest[:, -1]

            # Se entrena y clasifica
            clasificadorSL.entrenamiento(X_train, Y_train)
            predicciones = clasificadorSL.clasifica(X_test)

            # Se obtiene el error en las clasificaciones
            errores_particion.append(clasificadorSL.error(Y_test, predicciones))
            if self.matriz_confusion is None:
                self.matriz_confusion = clasificadorSL.matrizConfusion(Y_test, predicciones)
            else:
                self.matriz_confusion += clasificadorSL.matrizConfusion(Y_test, predicciones)


        self.matriz_confusion = self.matriz_confusion/(len(particionadoSL.listaParticiones))
        # Se devuelve la lista de errores en las clasificaciones
        return errores_particion

import matplotlib.pyplot as plt

import numpy as np
import copy

import numpy as np
import letritas

class Dataset():

    class Letra():
        def __init__(self, array_pixels, clase):
            self.array_pixels = array_pixels
            self.clase = clase

    def __init__(self, umbral=165, seed=None):
        self.datos_Bruto = []
        self.primera_Linea = []
        self.umbral = umbral
        self.permutacion = None
        self.tam_img = None
        self.datos = None
        
        self.seed = seed
        #random.seed(self.seed)
        np.random.seed(self.seed)
    
    def recortar(self, imagen, porcentaje_umbral_recorte=[0.05, 0.05, 0.05, 0.05]):
        '''
        Dada una imagen, se devuelve una imagen mas pequenya, ya recortada. Los cortes que
        se realizan son paralelos a los ejes y se realizan una vez que el porcentaje de
        pixeles negros encontrados en la columna o fila supera el "porcentajeUmbralRecorte"
        '''

        assert isinstance(porcentaje_umbral_recorte, (list, float)), 'Tipo de dato ' + str(type(porcentaje_umbral_recorte)) + '"porcentaje_umbral_recorte" incorrecto'

        ancho = imagen.shape[1]
        alto = imagen.shape[0]

        if isinstance(porcentaje_umbral_recorte, float):
            min_pixeles_arriba = alto * porcentaje_umbral_recorte
            min_pixeles_abajo = alto * porcentaje_umbral_recorte
            min_pixeles_izquierda = ancho * porcentaje_umbral_recorte
            min_pixeles_derecha = ancho * porcentaje_umbral_recorte
        else:
            min_pixeles_arriba = alto * porcentaje_umbral_recorte[0]
            min_pixeles_abajo = alto * porcentaje_umbral_recorte[1]
            min_pixeles_izquierda = ancho * porcentaje_umbral_recorte[2]
            min_pixeles_derecha = ancho * porcentaje_umbral_recorte[3]

        assert min_pixeles_arriba <= alto and min_pixeles_arriba >= 0, 'Valor de "min_pixeles_arriba" ' + str(min_pixeles_arriba) + ' incorrecto'
        assert min_pixeles_abajo <= alto and min_pixeles_abajo >= 0, 'Valor de "min_pixeles_abajo" ' + str(min_pixeles_abajo) + ' incorrecto'
        assert min_pixeles_izquierda <= ancho and min_pixeles_izquierda >= 0, 'Valor de "min_pixeles_izquierda" ' + str(min_pixeles_izquierda) + ' incorrecto'
        assert min_pixeles_derecha <= ancho and min_pixeles_derecha >= 0, 'Valor de "min_pixeles_derecha" ' + str(min_pixeles_derecha) + ' incorrecto'

        y_ini = 0
        y_fin = alto
        x_ini = 0
        x_fin = ancho


        y_ini = self.obtenerLimite(imagen, min_pixeles_arriba, reverse=False)
        y_fin = self.obtenerLimite(imagen, min_pixeles_abajo, max_tamanyo=alto,reverse=True)

        imagen_traspuesta = np.transpose(imagen)

        x_ini = self.obtenerLimite(imagen_traspuesta, min_pixeles_izquierda, reverse=False)
        x_fin = self.obtenerLimite(imagen_traspuesta, min_pixeles_derecha, max_tamanyo=ancho,reverse=True)

        if self.tam_img is None:
            self.tam_img = [x_fin - x_ini, y_fin - y_ini]
            return imagen[y_ini:y_fin, x_ini:x_fin]
        else:
            imagen_recortada = imagen[y_ini:y_fin, x_ini:x_fin]
            return self.redimensionarImagen(imagen_recortada, x_fin - x_ini, y_fin - y_ini)

    def obtenerLimite(self, imagen, min_pixeles, max_tamanyo=0, reverse=False):
        '''
        Dada una imagen, se corta por un lado a partir de que se supera "min_pixeles"
        si no se supera nunca se devuelve el mayor valor encontrado
        '''
        
        l_valores = np.zeros(shape=imagen.size)

        if reverse:
            # Se recorta la imagen a partir de que se supera min_pixeles
            for i_fila, _ in enumerate(imagen[0]):

                # Se obtiene el numero de pixeles que son negros en la imagen
                l_pixeles_negros = np.bincount(imagen[max_tamanyo - i_fila - 1 ])[:self.umbral]
                n_pixeles_negros = np.sum(l_pixeles_negros)

                # Si hay tantos o mas pixeles negros como el minimo numero
                # de pixeles se recorta (y_fin sera el numero de fila actual)
                l_valores[i_fila] += n_pixeles_negros
                if n_pixeles_negros >= min_pixeles:
                    return max_tamanyo - i_fila - 1
        else:
            # Se recorta la imagen a partir de que se supera min_pixeles
            for i_fila, fila in enumerate(imagen):

                # Se obtiene el numero de pixeles que son negros en la imagen
                l_pixeles_negros = np.bincount(fila)[:self.umbral]
                n_pixeles_negros = np.sum(l_pixeles_negros)

                # Si hay tantos o mas pixeles negros como el minimo numero
                # de pixeles se recorta (y_fin sera el numero de fila actual)
                l_valores[i_fila] += n_pixeles_negros
                if n_pixeles_negros >= min_pixeles:
                    return  i_fila

        return int(np.amax(l_valores))

    def redimensionarImagen(self, imagen, tam_h_actual, tam_v_actual):

        tam_h_actual -= 1
        tam_v_actual -= 1

        tam_v_final = self.tam_img[1]
        tam_h_final = self.tam_img[0]

        imagen_redimensionada = []
        for i in range(tam_v_final):
            imagen_redimensionada.append([])
            for j in range(tam_h_final):
                imagen_redimensionada[i].append(imagen[(round(i/tam_v_final*tam_v_actual))][(round(j/tam_h_final*tam_h_actual))])

        
        return np.array(imagen_redimensionada, dtype=np.int)

    def convertirPixeles(self, letra):
        # Comprueba si los valores RGB de los pixeles sobrepasan un umbral
        for i_elem, elem in enumerate(letra):
            if elem > self.umbral:
                letra[i_elem] = 0
            else:
                letra[i_elem] = 1   # La info que importa (negro)
        return letra

    def procesarDatos(self, celdas):
        # Obtiene el string con las clases desde parametros por defecto
        string_clases = letritas.parametros_por_defecto()[3]

        # Crea objetos Letra con su clase y su array de pixeles
        for i_celda_letra, celda_letra in enumerate(celdas):
            letra = self.Letra(celda_letra, i_celda_letra % len(string_clases))
            self.datos_Bruto.append(letra)

        return

    def procesarCuadraditos(self, tipo_atributo="cuadraditos", tamano=1,
                            n_pixeles_ancho=10, n_pixeles_alto=10,
                            porcentajeAgrupacion=0.01, solo_blanco_negro=False, random=True, hacer_recorte=True):
        # Comprueba el tipo de atributo (en funcion del mismo creara un dataset
        # de cuadraditos, filas o columnas)
        if tipo_atributo == "cuadraditos":

            # Convierte el array 1D devuelto por cuadraditos (usando la primera letra) en una fila
            # de un array 2D, de la que se extraen sus dimensiones
            primera_letra = self.datos_Bruto[0]
            if hacer_recorte:
                primera_letra.array_pixels = self.recortar(primera_letra.array_pixels)
            primer_dato = self.cuadraditos(primera_letra.array_pixels, primera_letra.clase, tamano)
            tam_fila_2d = primer_dato.shape[0]
            primer_dato = primer_dato.reshape(1, tam_fila_2d)

            # Crea el array 2D datos vacio con la dimension adecuada, y annade todas las letras a datos
            self.datos = np.empty((0, tam_fila_2d), dtype=int)
            self.datos = np.append(self.datos, primer_dato, axis=0)  # Primera letra

            for letra in self.datos_Bruto[1:]:  # Resto de letras
                if hacer_recorte:
                    letra.array_pixels = self.recortar(letra.array_pixels)

                dato = self.cuadraditos(letra.array_pixels, letra.clase, tamano)
                dato = dato.reshape(1, tam_fila_2d)

                self.datos = np.append(self.datos, dato, axis=0)
                #self.datos = np.append((self.datos, self.cuadraditos(letra.array_pixels, letra.clase, tamano)))


        elif tipo_atributo == "filas":

            # Convierte el array 1D devuelto por cuadraditosFilas (usando la primera letra) en una fila
            # de un array 2D, de la que se extraen sus dimensiones
            primera_letra = self.datos_Bruto[0]
            if hacer_recorte:
                primera_letra.array_pixels = self.recortar(primera_letra.array_pixels)
            primer_dato = self.cuadraditosFilas(primera_letra.array_pixels, primera_letra.clase, tamano)
            tam_fila_2d = primer_dato.shape[0]
            primer_dato = primer_dato.reshape(1, tam_fila_2d)

            # Crea el array 2D datos vacio con la dimension adecuada, y annade todas las letras a datos
            self.datos = np.empty((0, tam_fila_2d), dtype=int)
            self.datos = np.append(self.datos, primer_dato, axis=0)  # Primera letra

            for letra in self.datos_Bruto[1:]:  # Resto de letras
                if hacer_recorte:
                    letra.array_pixels = self.recortar(letra.array_pixels)

                dato = self.cuadraditosFilas(letra.array_pixels, letra.clase, tamano)
                dato = dato.reshape(1, tam_fila_2d)

                self.datos = np.append(self.datos, dato, axis=0)
                # self.datos = np.append(self.datos, self.cuadraditosFilas(letra.array_pixels, letra.clase, tamano))

        elif tipo_atributo == "columnas":
            # Convierte el array 1D devuelto por cuadraditosColumnas (usando la primera letra) en una fila
            # de un array 2D, de la que se extraen sus dimensiones
            primera_letra = self.datos_Bruto[0]
            if hacer_recorte:
                primera_letra.array_pixels = self.recortar(primera_letra.array_pixels)
            primer_dato = self.cuadraditosColumnas(primera_letra.array_pixels, primera_letra.clase, tamano)
            tam_fila_2d = primer_dato.shape[0]
            primer_dato = primer_dato.reshape(1, tam_fila_2d)

            # Crea el array 2D datos vacio con la dimension adecuada, y annade todas las letras a datos
            self.datos = np.empty((0, tam_fila_2d), dtype=int)
            self.datos = np.append(self.datos, primer_dato, axis=0)  # Primera letra

            for letra in self.datos_Bruto[1:]:  # Resto de letras
                if hacer_recorte:
                    letra.array_pixels = self.recortar(letra.array_pixels)

                dato = self.cuadraditosColumnas(letra.array_pixels, letra.clase, tamano)
                dato = dato.reshape(1, tam_fila_2d)

                self.datos = np.append(self.datos, dato, axis=0)
                #self.datos = np.append(self.datos, self.cuadraditosColumnas(letra.array_pixels, letra.clase, tamano))

        elif tipo_atributo == "random":
            
            primera_letra = self.datos_Bruto[0]

            if hacer_recorte:
                primera_letra.array_pixels = self.recortar(primera_letra.array_pixels)
            if solo_blanco_negro:
                primera_letra.array_pixels = self.todoBlancoNegro(primera_letra.array_pixels)

            l_cuadraditros = self.crearCuadraditos(0, primera_letra.array_pixels.shape[1], 0, primera_letra.array_pixels.shape[0],
                                                      n_pixeles_ancho=n_pixeles_ancho, n_pixeles_alto=n_pixeles_alto)
            self.datos = np.array([self.cuadraditosRandom(primera_letra.array_pixels, primera_letra.clase, l_cuadraditros,
                                             porcentajeAgrupacion=porcentajeAgrupacion, solo_blanco_negro=solo_blanco_negro, random=random)])
            

            for letra in self.datos_Bruto:  # Resto de letras
                if hacer_recorte:
                    letra.array_pixels = self.recortar(letra.array_pixels)

                if solo_blanco_negro:
                    letra.array_pixels = self.todoBlancoNegro(letra.array_pixels)

                l_cuadraditros = self.crearCuadraditos(0, letra.array_pixels.shape[1], 0, letra.array_pixels.shape[0],
                                                          n_pixeles_ancho=n_pixeles_ancho, n_pixeles_alto=n_pixeles_alto)
                dato = self.cuadraditosRandom(letra.array_pixels, letra.clase, l_cuadraditros,
                                                 porcentajeAgrupacion=porcentajeAgrupacion, solo_blanco_negro=solo_blanco_negro, random=random)
                self.datos = np.concatenate((self.datos, np.array([dato])))
        else:
            raise ValueError('Tipo de "tipo_atributo" ' + str(tipo_atributo) +' incorrecto.')

        return

    def cuadraditos(self, letra, clase, tamCuadrado=1):
        # Comprueba si los cuadraditos son bloques o pixeles
        if tamCuadrado == 1:
            
            # Aplana el array letra a 1D
            dimensiones = letra.shape[0] * letra.shape[1]
            letra = np.reshape(letra, dimensiones)
            
            # Codifica los pixeles de letra a 0s o 1s y annade la clase
            letra = self.convertirPixeles(letra)
            letra = np.append(letra, clase)
            
        else:
            # Crea un array 1D con los valores medios de cada bloque
            media = np.array([], dtype='int')

            # Recorre los bloques en los que se divide la imagen y calcula el valor medio
            # de cada uno. (Los "trozos" de imagen que no quepan en cuadrados se ignoran)
            for i in range(0, len(letra), tamCuadrado):
                if len(letra) - i < tamCuadrado:
                    break

                for j in range(0, letra.shape[1], tamCuadrado):
                    if letra.shape[1] - j < tamCuadrado:
                        break
                    media = np.append(media, int(np.mean(letra[i:(i+tamCuadrado), j:(j+tamCuadrado)])))

            # Codifica los pixeles de media a 0s o 1s y annade la clase
            letra = self.convertirPixeles(media)
            letra = np.append(letra, clase)

        return letra
        
    def cuadraditosFilas(self, letra, clase, tamFila=1):
        # Crea un array 1D con los valores medios de cada fila
        media= np.array([], dtype='int')

        # Recorre las filas en las que se divide la imagen y calcula el valor medio
        # de cada una. (Los "trozos" de imagen que no quepan en filas se ignoran)
        for i in range(0, len(letra), tamFila):
            if len(letra) - i < tamFila:
                break
            media = np.append(media, int(np.mean(letra[i:(i+tamFila), :])))

        # Codifica los pixeles de media a 0s o 1s y annade la clase
        letra = self.convertirPixeles(media)
        letra = np.append(letra, clase)

        return letra
        
    def cuadraditosColumnas(self, letra, clase, tamCol=1):
        # Crea un array 1D con los valores medios de cada columna
        media = np.array([], dtype='int')

        # Recorre las columnas en las que se divide la imagen y calcula el valor medio
        # de cada una. (Los "trozos" de imagen que no quepan en columnas se ignoran)
        for i in range(0, letra.shape[1], tamCol):
            if len(letra) - i < tamCol:
                break
            media = np.append(media, int(np.mean(letra[:, i:(i+tamCol)])))

        # Codifica los pixeles de media a 0s o 1s y annade la clase
        letra = self.convertirPixeles(media)
        letra = np.append(letra, clase)

        return letra
    
    def cuadraditosRandom(self, imagen_letra, clase, l_cuadraditros, porcentajeAgrupacion=0.0,
                          votacion="por_tamanyo", solo_blanco_negro=False, random=True):
        '''
        Dada una imagen y unas coordenadas que definen los cuadraditos en los que
        se ha dividido la imagen (las coordenadas estan en la lista l_cuadraditos).
        Se agrupan los cuadraditos segun "porcentajeAgrupacion" (100% significaria que
        todos los cuadraditos serian un atributo; por otra parte 0% significaria que cada
        cuadradito seria un atributo). En funcion de cual sea el color mayoritario en las
        agrupaciones que se realizaran de forma random, ese sera el valor del atributo
        del atributo en el diccionario a crear y devolver.
        Como hay cuadraditos mas grandes que otros se pueden configurar la votacion
        para que tengan mayor peso aquellos cuadraditos con mayor tamanyo
        '''

        # Se obtiene el numero de atributos
        num_cuadraditos = len(l_cuadraditros)
        
        # Si es la primera letra a transformar en datos todavia no hay
        # una permutacion de los indices de los cuadraditos a utilizar
        if self.permutacion is None:
            if random:
                self.permutacion = np.random.permutation(np.arange(num_cuadraditos))
            else:
                self.permutacion = np.arange(num_cuadraditos)#np.random.permutation(np.arange(num_cuadraditos))

        elif len(self.permutacion) != num_cuadraditos:
            print ("Algo ha ido mal el tamanyo de la permutacion no coincide con el numero de cuadraditos")
            return None

        # Se ajusta el "porcentajeAgrupacion" para evitar la division por cero
        # y para no repetir agrupaciones. Si porcentajeAgrupacion = 0.1
        # y num_cuadraditos = 6, habra 4 grupos de atributos repetidos
        if porcentajeAgrupacion <= 0 or num_cuadraditos < 1/porcentajeAgrupacion:
            porcentajeAgrupacion = 1/num_cuadraditos

        atributos = []        
        i_atributo = 0
        i_inferior = 0
        i_superior = porcentajeAgrupacion
        while i_superior <= 1:
            
            # Se obtiene la sublista de cuadraditos que conformaran un atributo
            l_cuadraditros_atributo = []
            for i in self.permutacion[round(num_cuadraditos*i_inferior):round(num_cuadraditos*i_superior)]:
                l_cuadraditros_atributo.append(l_cuadraditros[i])

            if solo_blanco_negro:
                atributos.append(self.votarBlancoNegro(imagen_letra, l_cuadraditros_atributo, votacion))
            else:
                atributos.append(self.mediaColores(imagen_letra, l_cuadraditros_atributo))

            
            i_atributo += 1
            i_inferior = i_superior
            i_superior += porcentajeAgrupacion
            i_superior = 1 if (i_superior < 1 and i_superior+porcentajeAgrupacion > 1) else i_superior
        
        return np.append(atributos, np.array(clase))
    
    def cuadraditosDiagonales(self):
        pass
    
    def patrones(self): #D
        pass
    
    def diferencia(self):
        pass
    
    def diferenciaPixel(self):
        datos = []
        for i, (fotos, letra) in enumerate(zip(np.column_stack(self.datos_Bruto), self.primera_Linea)):
            for foto in fotos:
                datos.append(np.append([letra - foto,i]))
        return datos
    
    def diferenciaPorcentaje(self):
        datos = []
        for i, (fotos, letra) in enumerate(zip(np.column_stack(self.datos_Bruto), self.primera_Linea)):
            sum_letra = np.sum(letra)/(len(letra)*len(letra[0]))
            for foto in fotos:
                sum_foto = np.sum(foto)/(len(foto)*len(foto[0]))
                datos.append([sum_letra - sum_foto,i])
        return datos
    
    def negros(self):
        pass

    def votarBlancoNegro(self, imagen, l_cuadraditros_atributo, votacion):
        '''
        Se vota si hay mas blancos, 
        '''

        # Por cada cuadradito "c", se incrementa el contador de votos del color en ese
        # cuadradito. Si se eligio la votacion por tamanyo cada cuadradito tiene tantos
        # votos como numero de pixeles en el cuadradito
        votacion_colores = {255: 0, 0: 0}
        for c in l_cuadraditros_atributo:
            color_mayoritario, n_votos_blanco, n_votos_negro = self.colorMayoritario(imagen,c[0][0],c[0][1],c[1][0],c[1][1])
            if votacion == "por_tamanyo":
                votacion_colores[255] += n_votos_blanco
                votacion_colores[0] += n_votos_negro
            else:
                votacion_colores[color_mayoritario] += 1

        if votacion_colores[255] >= votacion_colores[0]:
            return 0

        return 1

    def mediaColores(self, imagen, l_cuadraditros_atributo):
        '''
        Dada una imagen y una lista de coordenadas de inicio y fin de cuadraditos
        dentro de la imagen se devuelve la media de color de esos cuadraditos
        '''

        zonas_coordenadas = np.array([], dtype=np.int)

        for c in l_cuadraditros_atributo:
            # Se obtiene la zona de la imagen que esta dentro de las coordenas
            zona_coordenadas = imagen[c[1][0]:c[1][1], c[0][0]:c[0][1]]

            # Se cambia la forma del array para que este en 1D y asi poder contar cual es el elemento mayoritario
            zonas_coordenadas = np.append(zonas_coordenadas, np.reshape(zona_coordenadas, zona_coordenadas.size))

        return np.mean(zonas_coordenadas)

    def colorMayoritario(self, imagen, x_ini, x_fin, y_ini, y_fin):
        '''
        Dada una imagen y unas coordenadas de inicio y fin de la imagen se
        obtiene cual es el color mayoritario en el area de las coordenas
        y se devuelve dicho color junto con el resultado d ela votacion
        '''

        # Se obtiene la zona de la imagen que esta dentro de las coordenas
        zona_coordenadas = imagen[y_ini:y_fin, x_ini:x_fin]

        # Se cambia la forma del array para que este en 1D y asi poder contar cual es el elemento mayoritario
        zona_coordenadas = np.reshape(zona_coordenadas, zona_coordenadas.size)
        
        # Se realiza el conteo de elementos
        conteo = np.bincount(zona_coordenadas)

        # Si no hay pixeles blancos la longitud del conteo sera < 2, 
        # asi que se devuelve que hay 0 pixeles blancos
        if conteo.size < 2:
            return np.argmax(conteo), 0, conteo[0]
        else:
            return np.argmax(conteo), conteo[255], conteo[0]

    def crearCuadraditos(self, x_ini, x_fin, y_ini, y_fin, n_pixeles_ancho=None, n_pixeles_alto=None):
        '''
        Dada una imagen y unas coordenadas de inicio y fin de la imagen se
        crean divisiones, recortes, en la imagen que definiran las coordenadas
        de inicio y fin de cada cuadradito
        '''
        
        # None significa que se toma como tamanyo
        # el ancho (para n_pixeles_ancho) o el alto (para n_pixeles_alto)
        if n_pixeles_ancho is None:
            n_pixeles_ancho = x_fin - x_ini
        if n_pixeles_alto is None:
            n_pixeles_alto = y_fin - y_ini
        
        assert n_pixeles_ancho <= x_fin - x_ini and n_pixeles_alto >= 0, "Tamanyo de ancho incorrecto"
        assert n_pixeles_alto <= y_fin - y_ini and n_pixeles_ancho >= 0, "Tamanyo de alto incorrecto"

        l_cuadraditros = []

        # Se crean los limites de los cuadraditos
        for coordenada_y in range(y_ini, y_fin, n_pixeles_alto):
            for coordenada_x in range(x_ini, x_fin, n_pixeles_ancho):

                # Se obtiene las coordenadas x de inicio y fin del recorte del cuadradito
                if coordenada_x + n_pixeles_ancho > x_fin:
                    l_coordenadas = [(coordenada_x, x_fin)]
                else:
                    l_coordenadas = [(coordenada_x, coordenada_x + n_pixeles_ancho)]

                # Se obtiene las coordenadas y de inicio y fin del recorte del cuadradito
                if coordenada_y + n_pixeles_alto > y_fin:
                    l_coordenadas.append((coordenada_y, y_fin))
                else:
                    l_coordenadas.append((coordenada_y, coordenada_y + n_pixeles_alto))

                l_cuadraditros.append(l_coordenadas)


        return l_cuadraditros

    def quitaMargenes(self, corte_h, corte_v):
        '''
        Copia casi igual de extrae_cuadradito, pero
        solo elimina los margenes negros de una
        imagen y devuelve las coordenadas de la imagen
        '''
        alto = 200
        ancho = 150

        assert corte_h[1]-corte_h[0]+1 >= alto,  "Alto demasiado grande"
        assert corte_v[1]-corte_v[0]+1 >= ancho, "Ancho demasiado grande"
        
        margen_h = (corte_h[1] - corte_h[0] + 1 - alto)
        margen_v = (corte_v[1] - corte_v[0] + 1 - ancho)
        
        h0 = int(corte_h[0] +  margen_h // 2)
        h1 = int(corte_h[1] - (margen_h // 2 if margen_h % 2 == 0 else (1 + margen_h // 2)))
        v0 = int(corte_v[0] +  margen_v // 2)
        v1 = int(corte_v[1] - (margen_v // 2 if margen_v % 2 == 0 else (1 + margen_v // 2)))
        
        assert h1-h0+1==alto,  "H:{}-V:{}-C{}".format(corte_h, corte_v,[h0,h1])
        assert v1-v0+1==ancho, "H:{}-V:{}-C{}".format(corte_h, corte_v,[v0,v1])

        return h0, h1+1, v0, v1+1

    def mostrarImagen(self, imagen):
        '''
        Dada una imagen se muestra por pantalla
        '''
        plt.imshow(imagen, interpolation=None, cmap='gray')
        plt.show()

        return

    def mostrarImagenes(self, imagenes, columnas=10):
        '''
        Dado un grupo de imagenes se muestran por pantalla
        en forma de rejilla con numero de columnas "columnas"
        y el numero de filas que sean necesarias
        '''
        letritas.mostrar_imagenes2(imagenes, columnas)
        plt.show()

        return

    def todoBlancoNegro(self, imagen):
        '''
        Dada una imagen cambia todos los colores de la imagen
        por blancos y negros. El valor umbral donde da el color
        blanco viene dado por el valor "self.umbral"
        '''
        
        # Se crea una imagen en negro (todo ceros) donde se iran cambiando por blancos algunos pixeles
        imagen_aux = np.zeros(shape = (imagen.shape[0], imagen.shape[1]), dtype=np.int64)
        
        # Por cada pixel se comprueba que valor tiene
        for i_valor, valores_fila in enumerate(imagen):           
           for j_valor, valor in enumerate(valores_fila):
              
              # Si el valor del pixel es mayor a self.umbral decimos que es blanco
              if valor > self.umbral:
                 imagen_aux[i_valor][j_valor] += 255

        return imagen_aux

    def crearDataset(self, l_atributos, l_clases=None, ruta="ConjuntosDatos/", nombre="prueba.data"):

        with open(ruta+nombre, 'w') as fichero:

            # Se escribe el numero de datos que hay
            fichero.write(str(len(l_atributos))+"\n")

            # Se escribe el nombre de los atributos que hay
            atributos_nombre = ""
            num_atributos = len(l_atributos[0])
            # Si l_clases es None la clase ya estaba en l_atributos
            if l_clases is not None:
                for i in range(num_atributos):
                    atributos_nombre += "x" + str(i) + ","
                atributos_nombre += "class"
            else:
                for i in range(num_atributos-1):
                    atributos_nombre += "x" + str(i) + ","
                atributos_nombre += "class"
            fichero.write(atributos_nombre+"\n")

            # Se escribe el nombre de los atributos que hay
            tipo_atributos = ""
            # Si l_clases es None la clase ya estaba en l_atributos
            if l_clases is not None:
                for i in range(num_atributos):
                    tipo_atributos += "Continuo,"
                tipo_atributos += "Nominal"
            else:
                for i in range(num_atributos-1):
                    tipo_atributos += "Continuo,"
                tipo_atributos += "Nominal"
            fichero.write(tipo_atributos+"\n")

            # Se escriben los valores de cada dato y su clase
            for i_dato, atributos in enumerate(l_atributos):
                dato = ""
                for valor in atributos:
                    dato += str(valor) + ","

                # Si l_clases es None la clase ya estaba en
                # l_atributos asi que se quita la ultima coma
                if l_clases is not None:
                    dato += str(l_clases[i_dato])
                else:
                    dato = dato[:-1]
                fichero.write(dato+"\n")

        fichero.close()
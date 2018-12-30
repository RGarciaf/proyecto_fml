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

    def __init__(self, seed=None):
        self.datos_Bruto = []
        self.primera_Linea = []
        self.seed = seed
        self.datos = np.array([], dtype='int')
    
    def recortar(self): #D
        pass
    
    def convertirPixeles(self, letra, umbral=150):
        # Comprueba si los valores RGB de los pixeles sobrepasan un umbral
        for i_elem, elem in enumerate(letra):
            if elem > umbral:
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

    def procesarCuadraditos(self, tipo_atributo="cuadraditos", tamano=1):
        # Comprueba el tipo de atributo (en funcion del mismo creara un dataset
        # de cuadraditos, filas o columnas)
        if tipo_atributo == "cuadraditos":
            for letra in self.datos_Bruto:
                self.datos = np.append((self.datos, self.cuadraditos(letra.array_pixels, letra.clase, tamano)))

        elif tipo_atributo == "filas":
            for letra in self.datos_Bruto:
                self.datos = np.append(self.datos, self.cuadraditosFilas(letra.array_pixels, letra.clase, tamano))

        else:
            for letra in self.datos_Bruto:
                self.datos = np.append(self.datos, self.cuadraditosColumnas(letra.array_pixels, letra.clase, tamano))

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
    
    def cuadraditosRandom(self): #D
        pass
    
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

    def crearCuadraditos(self, x_ini, x_fin, y_ini, y_fin, n_pixeles_ancho=None, n_pixeles_alto=None):
        
        if n_pixeles_ancho is None:
            n_pixeles_ancho = x_fin - x_ini
        if n_pixeles_alto is None:
            n_pixeles_alto = y_fin - y_ini

        print (x_ini, x_fin, y_ini, y_fin, n_pixeles_ancho, n_pixeles_alto)
        
        l_cuadraditros = []

        for coordenada_x in range(x_ini, x_fin, n_pixeles_ancho):
            for coordenada_y in range(y_ini, y_fin, n_pixeles_alto):

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

    def mostrarImagen(self, imagen=None):
        plt.imshow(imagen, cmap='gray')
        plt.show()

    def mostrarImagenes(self, imagenes=None, columnas=10):
        letritas.mostrar_imagenes2(imagenes, columnas)
        plt.show()


    def todoBlancoNegro(self, imagen, umbral=165):
        imagen_aux = np.zeros(shape = (imagen.shape[0], imagen.shape[1]), dtype=np.int64)
        for i_valor, valores_fila in enumerate(imagen):
           for j_valor, valor in enumerate(valores_fila):
              if valor > umbral:
                 imagen_aux[i_valor][j_valor] += 255

        return imagen_aux

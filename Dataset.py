import matplotlib.pyplot as plt

import numpy as np
import copy

import numpy as np
import letritas

class Dataset():
    def __init__(self, seed=None):
        self.datos_Bruto = []
        self.primera_Linea = []
        self.seed = seed
    
    def recortar(self): #D
        pass
    
    def convertirPixeles(self, letra, umbral=150):
        
        for elem in letra:
            if elem > umbral:
                letra[elem] = 1
            else:
                letra[elem] = 0
        return letra
        

    def cuadraditos(self, letra, clase, tamCuadrado = 1):
        
        # Comprobar si los cuadraditos son bloques o pixeles
        if tamCuadrado == 1:
            
            # Aplana el array letra
            dimensiones = letra.shape[0] * letra.shape[1]
            
            letra = np.reshape(letra, dimensiones)
            
            # Codifica los pixeles de letra a 0s o 1s
            letra = self.convertirPixeles(letra)
            
            # Annade la clase
            letra = np.append(letra, clase)
            
        else:
            # TODO
            media = np.array()
            for i in range(len(letra), step = tamCuadrado):
                media = np.append(media, np.mean(letra[tamCuadrado*i:tamCuadrado*(i+1), 
                                                tamCuadrado*i:tamCuadrado*(i+1)]))
            letra = self.convertirPixeles(media)
            letra = np.append(letra, clase)
        return letra
        
    def cuadraditosFilas(self, letra, clase, tamFila = 1):
        media = np.array()
        for i in range(len(letra), step = tamFila):
            media = np.append(media, np.mean(letra[ : ,tamFila*i:tamFila*(i+1)]))
        letra = self.convertirPixeles(media)
        letra = np.append(letra, clase)
        return letra
        
    def cuadraditosColumnas(self, letra, clase, tamCol = 1):
        media = np.array()
        for i in range(len(letra), step = tamCol):
            media = np.append(media, np.mean(letra[ tamCol*i:tamCol*(i+1) , : ]))
        letra = self.convertirPixeles(media)
        letra = np.append(letra, clase)
        return letra
        
    
    def procesarCuadraditos(self):
       pass
    
    
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

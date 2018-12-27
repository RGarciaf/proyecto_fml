import numpy as np
import letritas

class Dataset():
     def __init__(self):
          self.datos_Bruto = []
          self.primera_Linea = []
     
     def recortar(self):
          pass
     
     def convertirPixeles(self, letra):
          
          for elem in letra:
               if elem > 150:
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
          
     
     def procesarCuadraditos(self)
    
     
     def cuadraditosRandom(self):
          pass
     
     def cuadraditosDiagonales(self):
          pass
     

     
     
     
     def patrones(self):
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
     
     
     
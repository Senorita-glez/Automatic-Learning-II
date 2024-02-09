import pandas as pd
import math, numpy

def BGD(dataset_caracteristicas,dataset_targets,iteraciones,f_aprendizaje,pesos_iniciales):
    pesos = []
    x_ij = []
    nuevos_pesos = []
    targets = dataset_targets.T.to_numpy()[0] # Pasamos el dataset a lista de numpy.
    ones = numpy.ones(len(dataset_caracteristicas)) # Creamos una lista de unos del tamaño de la cantidad de instancias.
    for j in range(iteraciones): # Por cada iteración
        if j == 0: pesos = pesos_iniciales # Si es la primera iteración, usemos los pesos iniciales.
        else: 
            pesos = numpy.copy(nuevos_pesos) # Los pesos que usaremos son los de la iteración anterior
            nuevos_pesos.clear()
        for i in range(len(pesos)): # Por cada peso que tengamos:
            if i == 0: # Si es w0, usamos a los dummies
                x_ij = ones
            else: # x_ij corresponde a todos los renglones y seleccionamos la columna en específico y lo hacemos lista.
                x_ij = dataset_caracteristicas.iloc[:,i-1].to_numpy()
            sum_parameter = ((x_ij * pesos[i]) - targets)*x_ij
            total_sum = numpy.sum(sum_parameter)
            nuevo_peso = pesos[i] - 2*(f_aprendizaje)*total_sum # Formulazo
            nuevos_pesos.append(round(nuevo_peso,6)) # Agregamos el peso calculado a nuestra lista de pesos

    return nuevos_pesos

def SGD(dataset_caracteristicas,dataset_targets,instancias,iteraciones,f_aprendizaje,pesos_iniciales):
    # Lo mismo que BGD pero con una sola instancia en lugar de todas.
    pesos = []
    x_ij = 0
    nuevos_pesos = []
    targets = dataset_targets.T.to_numpy()[0]
    for j in range(iteraciones):
        if j == 0: pesos = pesos_iniciales
        else: 
            pesos = numpy.copy(nuevos_pesos)
            nuevos_pesos.clear()
        for i in range(len(pesos)):
            if i == 0:
                x_ij = 1
            else:
                x_ij = dataset_caracteristicas.iloc[instancias.iloc[j,0],i-1]
            sum_parameter = ((x_ij * pesos[i]) - targets[instancias.iloc[j,0]])*x_ij
            nuevo_peso = pesos[i] - 2*(f_aprendizaje)*sum_parameter
            nuevos_pesos.append(round(nuevo_peso,4))
    
    return nuevos_pesos

if __name__ == '__main__':
    # Importamos los datasets
    casas = pd.read_csv('./casas.csv')
    instancias = pd.read_csv('./j.csv', header=None)

    # Separamos características y targets
    casas_caracteristicas = casas.iloc[:,:-1]
    casas_targets = casas.iloc[:,-1:]

    # Adquirimos parámetros necesarios
    print("\n\t*-------* BGD y SGD *-------*\n")
    iteraciones = int(input("Indica la cantidad de iteraciones: "))
    f_aprendizaje = float(input("Indica el factor de aprendizaje: "))
    print("Indica los pesos iniciales.")
    pesos_iniciales = []
    for i in range(len(casas_caracteristicas.columns)+1):
        aux = float(input("Peso " + str(i) + " (w"+str(i)+") : "))
        pesos_iniciales.append(aux)

    # Obtenemos los pesos calculados usando BGD y SGD
    bgd = BGD(casas_caracteristicas,casas_targets,iteraciones,f_aprendizaje,pesos_iniciales)
    sgd = SGD(casas_caracteristicas,casas_targets,instancias,iteraciones,f_aprendizaje,pesos_iniciales)

    print(f'\n{"*------"*8}\n')
    print(f'BGD: {bgd}\nSGD: {sgd}\n')
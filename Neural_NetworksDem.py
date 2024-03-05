import tensorflow as tf
print(tf.__version__)
import sofa
import polars as pl
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# Scikit-Learn
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.model_selection import KFold
# SciPy
from scipy.io import loadmat

# Tensorflow
from tensorflow.keras import models, regularizers, utils
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from itertools import product

import time
#********************************************************************
#                Demodulacion para Single Channel                   *
#********************************************************************
def demodulation_single(X_rx, X_tx, spacing,lista_neuronas, OSNR,n_splits,activations, database):
    """
    Demodulación para single channel

    :param X_rx: Diccionario con los datos recibidos (por espaciamiento)
    :param X_tx: Datos recibidos (Se lee previamente)
    :spacing: Espaciamiento actual (para escribir en archivo)
    :OSNR: Lista con los valores de OSNR para calcular su respectivo BER
    :n_splits: Numero de kFolds para la red
    :activations: Funciones de Activación para las capas de la red
    """
    folder = "/content/drive/MyDrive/Datos-Demodulation/Datos/Data_Base"
    #Variar el OSNR
    for i, snr in enumerate(X_rx):#Cambiar OSNR por X_rx si se desea calcular con todos los valores
        #snr=str(snr)+"dB"
        # Extraer información
        if snr not in database["single_ch"].keys():
            database["single_ch"][snr] = {}
            print(f"=================Iniciando {snr}=================")
        else:
            print(f"=================Continuando {snr}=================")
        #X_ch_norm = X_rx[snr].get("const_Y").flatten()
        X_ch_norm = np.array(X_rx[snr]['I'])+np.array(X_rx[snr]['Q'])*1j
        X_ch = sofa.mod_norm(X_ch_norm, 10) * X_ch_norm

        for neuron in lista_neuronas:
            if  (str(neuron)) not in database["single_ch"][snr].keys():
                DNN_BER_lista = []
                print(f"***********RED NEURONAL***********\n{neuron} Neuronas")
                # Sincronizar de las señales
                synced_X_tx = sofa.sync_signals(X_tx,  X_ch)[0]
                # Demodular señal transmitida
                y = sofa.demodulate(synced_X_tx, sofa.MOD_DICT)

                #Inicio asignacion de parametros para la red
                max_neurons = neuron
                layer_props_lst = [
                    {"units": max_neurons // (2**i), "activation": activation}
                    for i, activation in enumerate(activations)
                ]
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
                # Demodulación para redes
                t_inic = time.time()
                dem_DNN = sofa.demodulate_neural(
                    X_ch,
                    y,
                    layer_props_lst=layer_props_lst,
                    loss_fn=loss_fn,
                    n_splits = n_splits,
                    )
                demodulated, tests = dem_DNN
                #Calcular BER para cada KFold
                for demod, test in zip(demodulated, tests):
                    x_ch_dnn = np.argmax(demod, axis = 1)
                    k_BER = sofa.bit_error_rate(x_ch_dnn, test)
                    DNN_BER_lista.append(k_BER)
                print(f"Lista BER DNN: {DNN_BER_lista}")
                DNN_BER = np.mean(np.array(DNN_BER_lista))
                print(f"DNN mean BER=> {DNN_BER}")
                t_fin = time.time()
                print(f"Tiempo DNN: {t_fin - t_inic}\n")

                database["single_ch"][snr][neuron] = DNN_BER
                sofa.save_json(database, folder)
                print(f"Calculo con {neuron} NEURONAS terminado para OSNR {snr}dB ")
        print("Se han recorrido todas las neuronas")
    return database
#********************************************************************
#                        Demodulacion general                       *
#********************************************************************
def demodulation(X_rx, X_tx, spacing,lista_neuronas, OSNR,n_splits,activations, database, especific):
    """
    Demodulación con DNN para los casos que no son single channel

    :param X_rx: Diccionario con los datos recibidos (por espaciamiento)
    :param X_tx: Datos recibidos (Se lee previamente)
    :spacing: Espaciamiento actual (para escribir en archivo)
    :OSNR: Lista con los valores de OSNR para calcular su respectivo BER
    :n_splits: Numero de kFolds para la red
    :activations: Funciones de Activación para las capas de la red
    :database: Resultados en la base de datos
    :especific: Seleccionar snr específicos de la lista OSNR (true). Seleccionar todos
    """
    l_OSNR = X_rx
    if especific:
      l_OSNR = OSNR

    folder = "/content/drive/MyDrive/Datos-Demodulation/Datos/Data_Base"
    #Variar el OSNR
    for i, snr in enumerate(l_OSNR):#Cambiar OSNR por X_rx si se desea calcular con todos los valores
        # Extraer información
        if especific: #En database los datos vienen con una etiqueta específica
          snr=str(snr)+"dB"
        print(snr)
        if snr not in database[f"{spacing}GHz"].keys():
            database[f"{spacing}GHz"][snr] = {}
            print(f"=================Iniciando {snr}=================")
        else:
            print(f"=================Continuando {snr}=================")

        X_ch_norm = np.array(X_rx[snr]['I'])+np.array(X_rx[snr]['Q'])*1j
        X_ch = sofa.mod_norm(X_ch_norm, 10) * X_ch_norm
        sufix=""
        for act in activations:#Etiquetar por funcion de activación
          sufix=sufix+act[0]
        print(sufix)
        for neuron in lista_neuronas:
            if  (sufix+str(neuron)) not in database[f"{spacing}GHz"][snr].keys():
                DNN_BER_lista = []
                print(f"***********RED NEURONAL***********\n{neuron} Neuronas")
                # Sincronizar de las señales
                synced_X_tx = sofa.sync_signals(X_tx,  X_ch)[0]
                # Demodular señal transmitida
                y = sofa.demodulate(synced_X_tx, sofa.MOD_DICT)

                #Inicio asignacion de parametros para la red
                max_neurons = neuron
                layer_props_lst = [
                    {"units": max_neurons // (2**i), "activation": activation}
                    for i, activation in enumerate(activations)
                ]
                #Función de pérdida
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
                t_inic = time.time()#Medir el tiempo
                dem_DNN = sofa.demodulate_neural(
                    X_ch,
                    y,
                    layer_props_lst=layer_props_lst,
                    loss_fn=loss_fn,
                    n_splits = n_splits,
                    )
                demodulated, tests = dem_DNN
                #Calcular BER para cada KFold
                for demod, test in zip(demodulated, tests):
                    x_ch_dnn = np.argmax(demod, axis = 1)
                    k_BER = sofa.bit_error_rate(x_ch_dnn, test)
                    DNN_BER_lista.append(k_BER)

                print(f"Lista BER DNN: {DNN_BER_lista}")
                #Promedio del los BER encontrados
                DNN_BER = np.mean(np.array(DNN_BER_lista))
                print(f"DNN mean BER=> {DNN_BER}")
                t_fin = time.time()
                print(f"Tiempo DNN: {t_fin - t_inic}\n")

                database[f"{spacing}GHz"][snr][f"{sufix}{neuron}"] = DNN_BER
                sofa.save_json(database, folder)
                print(f"Calculo con {neuron} NEURONAS terminado para OSNR {snr}")
        print("Se han recorrido todas las neuronas")
    return database

#********************************************************************
#                           Leer Base de datos                      *
#********************************************************************
def read_data(folder_rx, ends):
    data = {}

    # Read root directory
    for folder in os.listdir(folder_rx):
        # Check name consistency for subdirectories
        if folder.endswith(ends):
            # Extract "pretty" part of the name
            spacing = folder[:-8]
            data[spacing] = {}

            # Read each data file
            for file in os.listdir(f"{folder_rx}/{folder}"):
                # Check name consistency for data files
                if file.find("consY") != -1:
                    # Extract "pretty" part of the name
                    osnr = file.split("_")[2][5:-4]

                    # Initialize if not created yet
                    if data[spacing].get(osnr) == None:
                        data[spacing][osnr] = {}
                    # Set data
                    csv_file_data = pl.read_csv(f"{folder_rx}/{folder}/{file}")
                    data[spacing][osnr] = csv_file_data
    return data
#********************************************************************
#                           Cargar datos                            *
#********************************************************************
folder_rx = "/content/drive/MyDrive/Datos-Demodulation/Datos/"
file_tx ="/content/drive/MyDrive/Datos-Demodulation/Datos/2x16QAM_16GBd.csv"
folder_json = "/content/drive/MyDrive/Datos-Demodulation/Datos/Data_Base"
# Datos transmitidos
X_tx_norm = pl.read_csv(file_tx)
X_tx_norm = np.array(X_tx_norm['I']) + np.array(X_tx_norm['Q'])*1j
X_tx = sofa.mod_norm(X_tx_norm, 10)*X_tx_norm
# Leer los datos recibidos
data = read_data(folder_rx,'spacing')
database = sofa.load_json(folder_json)

#********************************************************************
#                           Crear combinaciones                         *
#********************************************************************
osnr_lst = ["osnr", "wo_osnr"]
max_neurons = [str(2**n) for n in range(5, 11, 2)]
functs = ["relu", "tanh", "sigmoid"]
layers_n = [1, 2, 3]

combinations = [
    [list(subset) for subset in product(functs, repeat=n)] for n in layers_n
]

hidden_layers = [item for sublist in combinations for item in sublist]
classes_n = list(map(str, range(2, 6)))
#********************************************************************
#                           EJECUTAR MODELO                         *
#********************************************************************
spacings = [15,15.5,16,16.5,17,17.6,18]
for spacing in spacings:
    for layer in hidden_layers:
        X_rx = data[f'{spacing}GHz']
        print(f"Espaciamiento actual: {spacing}GHz")
        #demodulation(X_rx,X_tx, spacing,lista neuronas,[],4, ["relu","relu", "relu"],database, False)
        demodulation(X_rx=X_rx,
                     X_tx=X_tx,
                     spacing=spacing,
                     lista_neuronas=[32],
                     #Agregar OSNR específicos si se desea. (poner especific=True)
                     OSNR=[32],
                     n_splits=4,
                     activations=layer,
                     database=database,
                     #Indicar si se trabajan con valores de OSNR específico para evitar errores
                     especific=False
                     )
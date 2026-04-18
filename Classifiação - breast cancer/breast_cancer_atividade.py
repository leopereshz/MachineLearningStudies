import pandas as pd
import winsound

from sklearn.preprocessing import StandardScaler

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#carrega
previsores = pd.read_csv("entradas-breast.csv")
classe = pd.read_csv("saidas-breast.csv")

#transforma os valores dos atributos na mesma escala.
scaler= StandardScaler()
previsores = scaler.fit_transform(previsores)

def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units=16, activation='relu',kernel_initializer='uniform', input_dim = 30))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=16, activation='relu',kernel_initializer='uniform'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=1, activation='sigmoid'))
    
    otimizador = keras.optimizers.adam(lr=0.001, decay = 0.0001, clipvalue=0.3 )
    classificador.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'] )
    
    return classificador


#monta o classificador
classificador = KerasClassifier(build_fn = criarRede, epochs=100, batch_size=10)

#realiza o treinamento
resultados = cross_val_score(estimator= classificador, X=previsores, y=classe, cv=10, scoring='accuracy')

#calcula a média dos resultados
media = resultados.mean()
#calcula o desvio padrão para saber quão longe da media estão os valores.
desvio = resultados.std()

#gera um beep para avisar quando terminar
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 400  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
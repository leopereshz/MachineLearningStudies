#Breast cancer
#classificação
#objetivo é gerar um diagnóstico se o tumor é maligno ou benigno.

import pandas as pd
previsores = pd.read_csv("entradas-breast.csv")
classe = pd.read_csv("saidas-breast.csv")

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,classe, test_size=0.25)

import keras
from keras.models import Sequential
from keras.layers import Dense

classificador = Sequential()

#oculta 1
classificador.add(Dense(units=16, activation='relu',kernel_initializer='uniform', input_dim = 30))

#oculta 2
classificador.add(Dense(units=16, activation='relu',kernel_initializer='uniform'))

#saida
classificador.add(Dense(units=1, activation='sigmoid'))

#lr - learning rate.
#decay - diminui o learning rate ao longo das épocas.
#clipvalue - é um controle para evitar que a curva de loss aumente muito, então ele coloca uma trava e impede que ele suba.
otimizador = keras.optimizers.adam(lr=0.001, decay = 0.0001, clipvalue=0.5 )

classificador.compile(optimizer=otimizador, loss='binary_crossentropy',metrics=['binary_accuracy'])


classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100)

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes >0.5) #o retorno vem em probabilidade, então é necessário converter as previsões.

#calcular precisão

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste,previsoes)

#outra forma de extrair precisão e loss
resultado = classificador.evaluate(previsores_teste, classe_teste)
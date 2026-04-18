#Breast cancer
#classificação
#objetivo é gerar um diagnóstico se o tumor é maligno ou benigno.

import pandas as pd
previsores = pd.read_csv("entradas-breast.csv")
classe = pd.read_csv("saidas-breast.csv")

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,classe, test_size=0.25)

import keras
from keras.models import Sequential #modelo
from keras.layers import Dense      #camadas


classificador = Sequential()
#não é necessário especificar camada de entrada, apenas as ocultas.
#units = qtd neuronios internos = atributos de entrada (30) + classes na saidas (1) / 2  =1,5 => 16
#activation = função de ativação.
#kernel_initializer = inicializador de pesos.
#input_dim = quantos atributos tem na camada de entrada.
classificador.add(Dense(units=16, activation='relu',kernel_initializer='uniform', input_dim = 30))
#o keras por padrãa adiciona um neorônio de bias

#camada de saída
#units= quantos neurônios
classificador.add(Dense(units=1, activation='sigmoid'))

#optimizer = algoritmo para descida do gradiente
classificador.compile(optimizer='adam', loss='binary_crossentropy',metrics=['binary_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100)

#visualizar os pesos
pesos0 = classificador.layers[0].get_weights()
print(pesos0)
print(len(pesos0))

pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()




previsoes = classificador.predict(previsores_teste)
#o retorno vem em probabilidade, então é necessário converter as previsões.
previsoes = (previsoes >0.5)

#calcular precisão

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste,previsoes)

#outra forma de extrair precisão e loss
resultado = classificador.evaluate(previsores_teste, classe_teste)
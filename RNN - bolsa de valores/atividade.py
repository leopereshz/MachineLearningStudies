#Nas aulas anteriores nós fizemos as previsões dos preços das ações da Petrobras utilizando as ações
#de janeiro de 2013 até dezembro de 2017, com o intuito de prever os valores em janeiro de 2018.
#Em maio de 2018 tivemos a greve dos caminhoneiros no Brasil e as ações da Petrobras caíram bastante, 
#de aproximadamente R$ 25,00 para R$ 14,00! Com base nisso, 
#a ideia desta tarefa é verificar o comportamento da rede LSTM com essa grande variação no preço das ações.
#Siga as seguintes dicas:
#Você pode utilizar as ações de janeiro de 2013 até 25/05/2018 como base de dados de treinamento
# (dias antes da greve)
#Como base de testes, você pode utilizar os valores de 28/05/2018 até 22/06/2018.

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Base Treinamento
'''

base = pd.read_csv('petr4_treinamento_ex.csv')
base = base.dropna() #apaga registros onde valores não foram informados.

diasConsiderar = 90

#classe é o valor de abertura
base_treinamento = base.iloc[:, 1:2].values

#normalização. Transforma na escala 0 a 1.
normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

#Necessário definir o intervalo de tempo,
#pois os primeiros registros não podem ser previstos já que não tem dados anteriores para base.
#Então os atributos previsores serão os 90 últimos registros.
previsores = []
preco_real = []
for i in range(diasConsiderar, len(base_treinamento_normalizada)): #1242 é o total de registros
    previsores.append(base_treinamento_normalizada[i- diasConsiderar:i, 0]) #não considera o 90
    preco_real.append(base_treinamento_normalizada[i, 0]) #do 90 em diante

#transforma em numpay    
previsores, preco_real = np.array(previsores), np.array(preco_real)
#precisa colocar em um shape 3D
#dataset, quantidade de linhas, quantidade de colunas, 1 = indicador
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

'''
Rede
'''
regressor = Sequential()
#units = células de memória
#return_sequences = se vai ter mais camadas de LSTM para frente.
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 1)))
regressor.add(Dropout(0.3))

#nos testes deu resultados melhores com menos unidades de memória
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

#retira o return_sequences=True pois não tem mais LSTM pra frente
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 1, activation = 'linear'))

regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['mean_absolute_error'])
regressor.fit(previsores, preco_real, epochs = 100, batch_size = 32)

'''
Base Teste
'''

#formata data
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

base_teste = pd.read_csv('petr4_teste_ex.csv',parse_dates = ['Date'],date_parser = dateparse)
preco_real_teste = base_teste.iloc[:, 1:2].values
x_preco_real_teste = base_teste.iloc[:, 0:1].values
#converte o dataset para 90 dimensões, assim como o treinamento.

#concatena com a base de treinamento para pegar os 90 registros anteriores.
#axis = 0 é concatenação por coluna 
base_completa = pd.concat((base['Open'], base_teste['Open']), axis = 0)

#len(base_completa) - 1264, que é o somatório das 2 bases.
#len(base_teste) - quantidade de registros de teste.
# começa a buscar os registros da posição 1252 (as últimas 90 relacionadas a janeiro).
entradas = base_completa[len(base_completa) - len(base_teste) - diasConsiderar:].values
entradas = entradas.reshape(-1, 1)
#coloca na mesma escala do treinamento
entradas = normalizador.transform(entradas)

X_teste = []
for i in range(diasConsiderar, (diasConsiderar + len(preco_real_teste) ) ): #90 + 22
    X_teste.append(entradas[i-diasConsiderar :i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))

'''
Previsões
'''
previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

#verifica se a média dos preços ficam parecidas
previsoes.mean()
preco_real_teste.mean()

'''
Gráfico
'''    
plt.figure(figsize=(16,10))
plt.plot(x_preco_real_teste,preco_real_teste, color = 'red', label = 'Preço real')
plt.plot(x_preco_real_teste,previsoes, color = 'blue', label = 'Previsões')
plt.title('Previsão preço das ações')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()



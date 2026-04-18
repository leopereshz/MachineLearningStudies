
#No R existe nativamente o conjunto de dados women (mulheres), 
#com os atributos  height e weight (altura e peso, em polegadas e libras respectivamente)
#Usando regressão linear, preveja a altura de uma mulher com peso = 30


import pandas as pd   #para carregar a base
import numpy as np    #calculos matematicos
import matplotlib.pyplot as plt  #visualização de gráfico
from sklearn.linear_model import LinearRegression  #modelo de machine learning

#carrega dados
base = pd.read_csv('women.csv')
#remove a coluna não utilizada. axis=1 indica que é a coluna.
base = base.drop(['Unnamed: 0'], axis = 1)

#variavel independente
X_array = base.iloc[:, 1].values

#variável dependente.
y = base.iloc[:, 0].values

#calcula correlação
correlacao = np.corrcoef(X_array, y)

#treinar modelo
modelo = LinearRegression()
X_matriz = X_array.reshape(-1, 1)
modelo.fit(X_matriz, y)

inteseccao = modelo.intercept_

#coeficiente de determinação (R²)
R2 = modelo.score(X_matriz, y)

##GERA GRÁFICO
plt.scatter(X_matriz, y) #tanto faz se é array ou matriz.
#adiciona a linha com a previsão do modelo de treinamento. Nesse caso é obrigado usar a matriz.
plt.plot(X_matriz, modelo.predict(X_matriz), color = 'red')

#PREVISÃO TESTE - mulher com 30kg
previsao = modelo.intercept_ + modelo.coef_ * 30

#previsao = modelo.predict(30)
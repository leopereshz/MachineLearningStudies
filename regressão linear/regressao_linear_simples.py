import pandas as pd   #para carregar a base
import numpy as np    #calculos matematicos
import matplotlib.pyplot as plt  #visualização de gráfico
from sklearn.linear_model import LinearRegression  #modelo de machine learning

#carrega dados
base = pd.read_csv('cars.csv')
#remove a coluna não utilizada. axis=1 indica que é a coluna.
base = base.drop(['Unnamed: 0'], axis = 1)

#variavel independente
X_array = base.iloc[:, 1].values
#variável dependente.
y = base.iloc[:, 0].values

correlacao = np.corrcoef(X_array, y)

modelo = LinearRegression()
X_matriz = X_array.reshape(-1, 1) #Fit obriga que o parâmetro seja uma matriz 2D.
modelo.fit(X_matriz, y)

#Intesecção = onde a linha do eixo Y encontra-se com o eixo X =0.
inteseccao = modelo.intercept_

#Inclinação - a cada unidade que aumenta a variável independente (X), a variável de resposta (y) sobe o valor da inclinação.
inclinacao = modelo.coef_


##GERA GRÁFICO
plt.scatter(X_matriz, y) #tanto faz se é array ou matriz.
#adiciona a linha com a previsão do modelo de treinamento. Nesse caso é obrigado usar a matriz.
plt.plot(X_matriz, modelo.predict(X_matriz), color = 'red')


##PREVISÃO TESTE
# ex: distância 22 pés

#forma de cálculo 1. Fórmula y = ax + b
previsao = modelo.intercept_ + modelo.coef_ * 22
#forma de cálculo 2
previsao = modelo.predict(22)


#RESIDUAL - distância do ponto para a linha de regressão.
residual = modelo._residues     #resíduo de todos os registros.

#RESIDUAL INDIVIDUAL
# pip install yellowbrick
from yellowbrick.regressor import ResidualsPlot

#cria objeto para visualização
visualizador = ResidualsPlot(modelo)
visualizador.fit(X_matriz, y)
visualizador.poof()

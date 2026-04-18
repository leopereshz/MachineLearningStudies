##Dada as variáveis 
# X = 15,18,20,25,30,44
# Y = 240,255,270,283,300,310
#Calcule a coeficiente de correlação.

import pandas as pd   #para carregar a base
import numpy as np    #calculos matematicos
import matplotlib.pyplot as plt  #visualização de gráfico
from sklearn.linear_model import LinearRegression  #modelo de machine learning

#variavel independente
X_array = np.array([15,18,20,25,30,44])

#variável dependente.
y = np.array([240,255,270,283,300,310])

#calcula correlação
correlacao = np.corrcoef(X_array, y)

#previsoes
modelo = LinearRegression()
X_matriz = X_array.reshape(-1, 1)
modelo.fit(X_matriz, y)

#coeficiente de determinação (R²)
R2 = modelo.score(X_matriz, y)

##GERA GRÁFICO
plt.scatter(X_matriz, y) #tanto faz se é array ou matriz.
#adiciona a linha com a previsão do modelo de treinamento. Nesse caso é obrigado usar a matriz.
plt.plot(X_matriz, modelo.predict(X_matriz), color = 'red')
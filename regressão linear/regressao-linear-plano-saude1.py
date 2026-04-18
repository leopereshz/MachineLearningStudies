import pandas as pd

base = pd.read_csv('plano_saude.csv')

X = base.iloc[:, 0].values #idade
y = base.iloc[:, 1].values #preço

import numpy as np
#indica o quanto uma variável está próximo de outra, ou seja, quanto mais uma aumenta, a outra aumenta também
correlacao = np.corrcoef(X, y)
#se a correlação for:
# 1 - perfeita
# 0,7-forte
# 0,5-moderada
# 0,25-fraca
# 0 - inexistente
#-0,25-fraca
#-0,5-moderada
#-0,7-forte
#-1 - perfeita
#a distancia pode ser positiva ou negativa, pois no gráfico o erro pode ser para cima ou para baixo.

#transforma o vetor em matriz, pois o scikit learn exige.
X = X.reshape(-1,1)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# b0 - visualizar
regressor.intercept_

# b1 - visualizar
regressor.coef_

#plotar gráfico
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color = 'red')
plt.title ("Regressão linear simples")
plt.xlabel("Idade")
plt.ylabel("Custo")

# previsão pessoa com 40 anos
previsao1 = regressor.intercept_ + regressor.coef_ * 40 #calcular na mão
previsao2 = regressor.predict(40) #usando predict

score = regressor.score(X,y)


#instalar biblioteca de graficos
#pip install yellowbreak
from yellowbrick.regressor import ResidualsPlot
visualizador = ResidualsPlot(regressor)
visualizador.fit(X, y)
visualizador.poof()

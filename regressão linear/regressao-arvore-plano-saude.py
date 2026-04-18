import pandas as pd

base = pd.read_csv('plano_saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X,y)
score = regressor.score(X,y)

import matplotlib.pyplot as plt
plt.scatter(X,y)
plt.plot(X,reressor.predict(X), color='red')
plt.title('Regressão com árvores')
plt.xlabel('Idade')
plt.ylabel('Custo')

#os dados acima deu um score perfeito porque a base tem poucos dados.

import numpy as np
X_teste = np.arange(min(X), max(X), 0.1) #popula a coleção de testes com todas as possibilidades. 0.1 é a diferença entre cada valor.
X_teste = X_teste.reshape(-1,1) #transforma em matriz
plt.scatter(X,y)
plt.plot(X_teste,regressor.predict(X_teste),color='red')
plt.xlabel('Idade')
plt.ylabel('Custo')

#o modelo de árvore de decisão não é contínuo nem linear.
#os resultados no gráfico são como uma escada, onde cada degrau representa os splits.

regressor.predict(40)
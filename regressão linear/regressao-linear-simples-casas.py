import pandas as pd

base = pd.read_csv('house_prices.csv')

X = base.iloc[:, 5:6].values #m2 - a coluna 6 é só para forçar a criação da matriz, não tem uso.
y = base.iloc[:, 2].values	 #valor

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)
score = regressor.score(X_treinamento, y_treinamento)

##tenta prever o valor com base no tamanho da casa.
import matplotlib.pyplot as plt
plt.scatter(X_treinamento, y_treinamento)
plt.plot(X_treinamento, regressor.predict(X_treinamento), color = 'red')
previsoes = regressor.predict(X_teste)
#diferença entre a base e a previsão
resultado = abs(y_teste - previsoes)
resultado.mean() #media de erro

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_teste, previsoes)	#equivalente ao resultado = abs(y_teste - previsoes)
mse = mean_squared_error(y_teste, previsoes)

plt.scatter(X_teste, y_teste)
plt.plot(X_teste, regressor.predict(X_teste), color = 'red')

regressor.score(X_teste, y_teste)

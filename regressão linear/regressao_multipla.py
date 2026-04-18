import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#carrega dados
base = pd.read_csv('mt_cars.csv')
base = base.drop(['Unnamed: 0'], axis = 1)

#separa atributos independentes e dependentes.
X_array = base.iloc[:, 2].values
y = base.iloc[:, 0].values

correlacao = np.corrcoef(X_array, y)

#transforma em matriz para utilizar o LinearRegression
X_matriz = X_array.reshape(-1, 1) #-1 e 1 - não mexe nas linhas e adiciona coluna

modelo = LinearRegression()
modelo.fit(X_matriz, y)

#Intesecção = onde a linha do eixo Y encontra-se com o eixo X =0.
inteseccao = modelo.intercept_

#Inclinação - a cada unidade que aumenta a variável independente (X), a variável de resposta (y) sobe o valor da inclinação.
inclinacao =modelo.coef_

#coeficiente de determinação
#Mostra o quanto o modelo consegue explicar os valores.
#Recomendável apenas quando tem 1 atrributo independente.
R2 = modelo.score(X_matriz, y)

##########
#quando tem mais atributos independentes, recomenda-se o coeficiente ajustado.

previsoes = modelo.predict(X_matriz)

#gerar modelo no python como se estivesse escrevendo na linguagem R.
import statsmodels.formula.api as sm

modelo_ajustado = sm.ols(formula = 'mpg ~ disp', data = base)
#ols = ordinary list squares (um tipo de regressão)
#formula  = mpg (atributo que queremos prever), disp (atributo independente), data (base de dados originais, sem precisar separar x e y)

modelo_treinado = modelo_ajustado.fit()
#resumo
modelo_treinado.summary()
#no resumo que gerar, procurar pelo Adj. R-squared.

#######
#visualizar grafico
plt.scatter(X_matriz, y)
plt.plot(X_matriz, previsoes, color = 'red')


#########
#previsão
valor_prever = np.array([200])
valor_prever = valor_prever.reshape(1,-1)
previsao = modelo.predict(valor_prever)  #ex: veiculo com 200 polegadas cubidcas (mpg).


##################################
#Regressão linear múltipla
##################################

X1 = base.iloc[:, 1:4].values
y1 = base.iloc[:, 0].values

modelo2 = LinearRegression()
modelo2.fit(X1, y1)

R2 = modelo2.score(X1, y1)

#R ajustado
modelo_ajustado2 = sm.ols(formula = 'mpg ~ cyl + disp + hp', data = base)
#mgp - valor a prever.
# cyl + disp+ hp = atributos independentes

modelo_treinado2 = modelo_ajustado2.fit()
modelo_treinado2.summary()

####
#previsão
novo = np.array([4, 200, 100]) 
novo = novo.reshape(1, -1)   # 1 e -1 = adiciona uma linha e não mexe nas colunas
modelo2.predict(novo)


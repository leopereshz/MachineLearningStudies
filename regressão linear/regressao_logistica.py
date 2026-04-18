import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

#carrega base
base = pd.read_csv('Eleicao.csv', sep = ';')

#gera gráfico
plt.scatter(base.DESPESAS, base.SITUACAO)
base.describe()

#correlação
correlacao = np.corrcoef(base.DESPESAS, base.SITUACAO)

X = base.iloc[:, 2].values
X = X[:, np.newaxis] #mesmo que reshape(1 -1)
y = base.iloc[:, 1].values

modelo = LogisticRegression()
modelo.fit(X, y)

#Inclinação - a cada unidade que aumenta a variável independente (X), a variável de resposta (y) sobe o valor da inclinação.
inclinacao = modelo.coef_

#Intesecção = onde a linha do eixo Y encontra-se com o eixo X =0.
inteseccao = modelo.intercept_

plt.scatter(X, y)
#gera 100 numeros aleatórios de 10 a 3000 
X_teste = np.linspace(10, 3000, 100)

def model(x):
    return 1 / (1 + np.exp(-x)) #sigmoide

r = model(X_teste * modelo.coef_ + modelo.intercept_).ravel() #rabel() transforma numpy em vetor

plt.plot(X_teste, r, color = 'red')

#####
#previsoes

#carrega dados
base_previsoes = pd.read_csv('NovosCandidatos.csv', sep = ';')

despesas = base_previsoes.iloc[:, 1].values
despesas = despesas.reshape(-1, 1)

#prever
previsoes_teste = modelo.predict(despesas)

#visualziar os resultados por candidato.
base_previsoes = np.column_stack((base_previsoes, previsoes_teste)) 
#executar base_previsoes no console para ver os resultados
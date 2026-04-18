#pip install skcikit-fuzzy

from sklearn import datasets
import numpy as np
from sklearn.metrics import confusion_matrix
import skfuzzy

iris = datasets.load_iris()

#necessario transformar os dados do iris, transpor linha e coluna.
resultado = skfuzzy.cmeans(data = iris.data.T, c = 3, m = 2, error = 0.005, maxiter = 1000, init = None)
previsoes_porcentagem = resultado[1]

previsoes_porcentagem[0][0]
previsoes_porcentagem[1][0]
previsoes_porcentagem[2][0]

#pega o maior valor por coluna.
previsoes = previsoes_porcentagem.argmax(axis = 0)

resultados = confusion_matrix(iris.target, previsoes)
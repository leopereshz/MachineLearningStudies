#detectar de forma mais automática os outliers.
#rodando toda a base ao invés de comparar de 2 em 2 atributos.

import pandas as pd

base = pd.read_csv('credit_data.csv')
base = base.dropna()

#python outlier detection.
#pip install pyod
from pyod.models.knn import KNN
detector = KNN() #existem vários algoritmos para detectar outliers, na documentação da biblioteca pyod existe uma descrição detalhada de cada um.
detector.fit(base.iloc[:,1:4])

previsoes = detector.labels_ #0 = não outlier, 1= outlier.
confianca_previsoes = detector.decision_scores_ #confiança de cada registro, com base na distância euclidiana.

#adiciona uma coluna que identifica se é outlier ou não
outliers = []
for i in range(len(previsoes)):
    #print(previsoes[i])
    if previsoes[i] == 1:
        outliers.append(i)

#filtra a lista de outliers.      
lista_outliers = base.iloc[outliers, :]
    
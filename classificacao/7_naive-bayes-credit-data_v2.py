import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imputer = imputer.fit(previsores[:, 1:4])

previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()

#constrói tabela de probabilidades
classificador.fit(previsores_treinamento, classe_treinamento)
#submete a base de testes ao classificador para prever os resultados
previsoes = classificador.predict(previsores_teste)

######################################################
#compara resultados
######################################################
from sklearn.metrics import confusion_matrix, accuracy_score
#calcula % de acerto
precisao = accuracy_score(classe_teste, previsoes)
#matriz de confusão. Relaciona cada valor possível quantos registros foram 
#classificados corretamente e incorretamente em relação a base de teste.
matriz = confusion_matrix(classe_teste, previsoes)

#matriz de confusão melhorada
from yellowbrick.classifier import ConfusionMatrix
v = ConfusionMatrix(GaussianNB())
v.fit(previsores_treinamento,classe_treinamento)
v.score(previsores_teste,classe_teste)
v.poof()


##############################
#BASE LINE CLASSIFIER
#GERA A PROBABILIDADE ATRAVÉS DE UM COUNT SIMPLES, PARA VOCÊ TESTAR SE VALE A PENA FAZER TODO O PROCESSAMENTO ACIMA
#pra achar o % tem que dividir o resultado pelo total de registros da amostra.
##############################
import collections
collections.counter(classe_teste)


import pandas as pd

base = pd.read_csv('0_census.csv')

previsores	= base.iloc[:, 0:14].values
classe		= base.iloc[:, 14].values

              
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])

onehotencoder = OneHotEncoder(categorical_features = [1,3,5,6,7,8,9,13])
previsores = onehotencoder.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#separa a base de treinamento da base de teste.
from sklearn.cross_validation import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

#gera tabela de probabilidade
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)

#gera estimativa de resultado para cada registro.
previsoes = classificador.predict(previsores_teste)

#calcula a precisão
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
#matriz resumo de quantidades certas e erradas.
matriz = confusion_matrix(classe_teste, previsoes)



##############################
#BASE LINE CLASSIFIER
#GERA A PROBABILIDADE ATRAVÉS DE UM COUNT SIMPLES, PARA VOCÊ TESTAR SE VALE A PENA FAZER TODO O PROCESSAMENTO ACIMA
#pra achar o % tem que dividir o resultado pelo total de registros da amostra.
##############################
import collections
collections.counter(classe_teste)

#Regressão múltiplas saídas.
#vendas no japão, vendas na América do norte e Europa.

import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

base = pd.read_csv('games.csv')
base = base.drop('Other_Sales', axis = 1)
base = base.drop('Global_Sales', axis = 1)
base = base.drop('Developer', axis = 1)

base = base.dropna(axis = 0)
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

#remove o nome, mas transfere para outra coleção.
base['Name'].value_counts()
nome_jogos = base.Name
base = base.drop('Name', axis = 1)




previsores = base.iloc[:, [0,1,2,3,7,8,9,10,11]].values
#os 3 abaixo são as classes
venda_na = base.iloc[:, 4].values
venda_eu = base.iloc[:, 5].values
venda_jp = base.iloc[:, 6].values




from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 8])

# s 1 0
# r 0 1
onehotencoder = OneHotEncoder(categorical_features = [0,2,3,8])
previsores = onehotencoder.fit_transform(previsores).toarray()


#segundo o professor é uma forma diferente de fazer a mesma coisa.
camada_entrada = Input(shape=(61,)) #tem 61 colunas.
camada_oculta1 = Dense(units = 32, activation = 'sigmoid')(camada_entrada)
camada_oculta2 = Dense(units = 32, activation = 'sigmoid')(camada_oculta1)
camada_saida1 = Dense(units = 1, activation = 'linear')(camada_oculta2)  #America do norte
camada_saida2 = Dense(units = 1, activation = 'linear')(camada_oculta2)	 #europa
camada_saida3 = Dense(units = 1, activation = 'linear')(camada_oculta2)  #japão

#é o que monta o modelo propriamente
regressor = Model(inputs = camada_entrada, outputs = [camada_saida1, camada_saida2, camada_saida3])

regressor.compile(optimizer = 'adam', loss = 'mse')

#a classe são os 3 atributos
regressor.fit(previsores, [venda_na, venda_eu, venda_jp], epochs = 5000, batch_size = 100)

previsao_na, previsao_eu, previsao_jp = regressor.predict(previsores)






























import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

#foi necessário passar o encoding.
base = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')

#agrupa por nome e mostra a quantidade de vezes que ele se repete no dataset.
base['name'].value_counts()
base['seller'].value_counts()
base['offerType'].value_counts()
#no caso as 3 colunas tinham muita variabilidade e optou-se por excluir.


#apaga atributos inúteis
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)
base = base.drop('name', axis = 1)
base = base.drop('seller', axis = 1)
base = base.drop('offerType', axis = 1)


#Corrige valores inconsistentes

i1 = base.loc[base.price <= 10]	#identifica os registros com valor menor que 10. 
base.price.mean()	#média de preço.
#filtra apenas os registros com valor maior que especificado.
base = base[base.price > 10]
i2 = base.loc[base.price > 350000]
base = base.loc[base.price < 350000]


#identifica valores nulos
base.loc[pd.isnull(base['vehicleType'])] #iidentifica os registros nulos
base['vehicleType'].value_counts() # limousine é o registro que mais aparece
base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() # manuell
base.loc[pd.isnull(base['model'])]
base['model'].value_counts() # golf
base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts() # benzin = gasolina
base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts() # nein

#substitui valores onde tinha null.
valores = {'vehicleType': 'limousine', 'gearbox': 'manuell',
           'model': 'golf', 'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}
base = base.fillna(value = valores)





previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 10] = labelencoder_previsores.fit_transform(previsores[:, 10])

# 0 = 0 0 0
# 2 = 0 1 0
# 3 = 0 0 1
onehotencoder = OneHotEncoder(categorical_features = [0,1,3,5,8,9,10])
previsores = onehotencoder.fit_transform(previsores).toarray()


#neste exemplo ele não separou as bases de treinamento e testes

regressor = Sequential()
regressor.add(Dense(units = 158, activation = 'relu', input_dim = 316))
regressor.add(Dense(units = 158, activation = 'relu'))
regressor.add(Dense(units = 1, activation = 'linear'))  #como está tentando prever o valor do carro, 
#não é possível usar uma função de ativação para transformar o valor. Então linear faz absolutamente nada.

regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['mean_absolute_error'])
regressor.fit(previsores, preco_real, batch_size = 300, epochs = 100)


previsoes = regressor.predict(previsores)
preco_real.mean()
previsoes.mean()

























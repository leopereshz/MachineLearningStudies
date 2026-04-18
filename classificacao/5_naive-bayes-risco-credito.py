# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:36:39 2017

@author: Jones
"""
import pandas as pd

base = pd.read_csv('0_risco_credito.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

#####################################################
#TRANSFORMAÇÕES
#NAIVE BAYES PRECISA CONVERTER
#####################################################
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])

#Nesse exemplo não precisa usar OneHotEncoder porque todos os valores já estão em 0 e 1.

#####################################################
#TREINAMENTO COM NAIVE BAYES
#####################################################

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)

#####################################################
#TESTAR PREVISÃO
#####################################################

# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
resultado = classificador.predict([[0,0,1,2], [3, 0, 0, 0]])

#####################################################
#MOSTRAR RESULTADOS
#####################################################

print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_) #PROBABILIDADES Á PRIORI


import pandas as pd

base = pd.read_csv('0_census.csv')

#divide o dataset em atributos previsores e a classe (resultados)
previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values

#####################################################
#TRANSFORMAR DADOS CATEGÓRICOS EM DISCRETOS
#transforma em números, ex: 0,1,2,3,4,5
#####################################################

from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()

#exemplo de transformação de uma coluna. Começa a contar no zero.
#label = labelencoder_previsores.fit_transform(previsores[:,1])

#aplicar em todos que são categóricos
previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelencoder_previsores.fit_transform(previsores[:,5])
previsores[:,6] = labelencoder_previsores.fit_transform(previsores[:,6])
previsores[:,7] = labelencoder_previsores.fit_transform(previsores[:,7])
previsores[:,8] = labelencoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelencoder_previsores.fit_transform(previsores[:,9])
previsores[:,13] = labelencoder_previsores.fit_transform(previsores[:,13])

#transforma a coluna income em categóricos
labelencoderclasse = LabelEncoder()
classe = labelencoderclasse.fit_transform(classe)

#####################################################
#TRANSFORMAR DADOS CATEGÓRICOS EM DISCRETOS
#UTILIZA VARIÁVEIS DUMMY
#transforma em matrizes de 0 e 1
#precisa executar o labelencoder antes
#####################################################

from sklearn.preprocessing import OneHotEncoder
#inicializa objeto.
oneHotEncoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
#cria variáveis dummy para os atributos selecionados no objeto.
previsores = oneHotEncoder.fit_transform(previsores).toarray()


#####################################################
#ESCALONAMENTO
#####################################################

#escalonamento dos valores
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)








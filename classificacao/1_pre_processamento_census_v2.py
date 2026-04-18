import pandas as pd

base = pd.read_csv('0_census.csv')

#divide o dataset em atributos previsores e a classe (resultados)
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

#####################################################
#TRANSFORMAR DADOS CATEGÓRICOS EM DISCRETOS
#transforma em números, ex: 0,1,2,3,4,5

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')

previsores = onehotencorder.fit_transform(previsores).toarray()

#transforma a coluna income em categóricos
labelencorder_classe = LabelEncoder()
classe = labelencorder_classe.fit_transform(classe)

#ESCALONAMENTO
#####################################################
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

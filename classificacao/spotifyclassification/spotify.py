import pandas as pd
base = pd.read_csv('spotifyclassification/data.csv')
base.drop('id',1,inplace=True)
base.drop('song_title',1,inplace=True)
previsores = base.iloc[:,0:15].values
classe = base.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
previsores[:,14] = labelencoder.fit_transform(previsores[:,14])


#onehotencoder = OneHotEncoder(categorical_features = [2,5,11])
#previsores = onehotencoder.fit_transform(previsores).toarray()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.cross_validation import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)


previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)
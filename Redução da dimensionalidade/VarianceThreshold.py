import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold

dataset = pd.read_csv('ad.data', header=None)
#visualizar primeiros registros
dataset.head()

#1558 é a última coluna
dataset[1558].unique()

x = dataset.iloc[:0:1558].values
y = dataset.iloc[:, 1558].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 0)

model1 = GaussianNB()
model1.fit(x_train, y_train)

prediction1 = model1.predict(x_test)

accuracy_score(y_test, prediction1) #0.7754

#SELEÇÃO DE ATRIBUTOS

#características com variância MENOR que o treshold serão removidas.
selection = VarianceThreshold(treshold=0.159) #valor definido aleatoriamente

x_new = selection.fit_transform(x)

selection.variances_ 

#como tem 1557 atributos, para visualizar todos é necessário usar um for
for i in range(len(selection.variances_ )):
	print(selection.variances_[i])

#filtra apenas colunas com threshold maior
indexes = np.where(selection.variances_ > 0.159)

#faz um novo treinamento só com as colunas importantes

x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.3, random_state = 0)

model2 = GaussianNB()
model2.fit(x_train, y_train)
prediction2 = model2.predict(x_test)
accuracy_score(y_test, prediction2) #0.9227

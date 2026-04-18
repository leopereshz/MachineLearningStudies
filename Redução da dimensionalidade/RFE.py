#Recursive Feature Elimination
#Ele começa treinando o modelo com todos os atributos
#e vai aos poucos testando eliminar os atributos com menos importância
#e treina novamente, até que não consiga mais melhorar o modelo.

from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE

data = load_iris()
X = data.data
y = data.target
# 
model = LinearSVC()
rfe = RFE(model, step=1).fit(X, y)

selection of attributes
print(rfe.support_)
print(rfe.ranking_)
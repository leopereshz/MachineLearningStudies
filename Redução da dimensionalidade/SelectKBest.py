#ele seleciona automaticamente os K atributos mais relevantes

from sklearn.datasets import load_iris
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

data = load_iris()
X = data.data
y = data.target
X = SelectKBest(chi2, k=2).fit_transform(X, y)
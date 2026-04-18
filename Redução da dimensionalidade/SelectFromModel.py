#pode ser utilizado com qualquer algoritmo do tipo arvore ou Ensambla.
#elimina os atributos que não satisfaçam o threshold minimo que você definir. 

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

iris = load_iris()
X, y = iris.data, iris.target
X.shape
#(150, 4)
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)
clf.feature_importances_  
#array([ 0.04...,  0.05...,  0.4...,  0.4...])
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X_new.shape               
#(150, 2)
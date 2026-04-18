import pandas as pd
iris = pd.read_csv("iris.csv")

previsores = iris.iloc[:, 0:4].values

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'random')
kmeans.fit(previsores)

#identificar centóides
kmeans.cluster_centers_

#tabela de distâncias
distance = kmeans.fit_transform(previsores)

#como encontrar o valor ideal de clusters (grupos)?
######################################################
from sklearn.cluster import KMeans
wcss = []
 
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'random')
    kmeans.fit(X)
    print i,kmeans.inertia_
    wcss.append(kmeans.inertia_)  
plt.plot(range(1, 11), wcss)
plt.title('O Metodo Elbow')
plt.xlabel('Numero de Clusters')
plt.ylabel('WSS') #within cluster sum of squares
plt.show()

#testar novos valores
##########################
data = [
        [ 4.12, 3.4, 1.6, 0.7],
        [ 5.2, 5.8, 5.2, 6.7],
        [ 3.1, 3.5, 3.3, 3.0]
    ]
kmeans.predict(data)

#plotando resultados
###########################################
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:,1], s = 100, c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'red',label = 'Centroids')
plt.title('Iris Clusters and Centroids')
plt.xlabel('SepalLength')
plt.ylabel('SepalWidth')
plt.legend()

plt.show()



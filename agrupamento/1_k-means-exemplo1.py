import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#popular valores
x=[20,  27,  21,  37,  46, 53, 55,  47,  52,  32,  39,  41,  39,  48,  48]  
y=[1000,1200,2900,1850,900,950,2000,2100,3000,5900,4100,5100,7000,5000,6500]  
plt.scatter(x,y) #previsualizar

base = np.array([[20,1000],[27,1200],[21,2900],[37,1850],[46,900],
                 [53,950],[55,2000],[47,2100],[52,3000],[32,5900],
                 [39,4100],[41,5100],[39,7000],[48,5000],[48,6500]])

scaler = StandardScaler()
base = scaler.fit_transform(base)

kmeans = KMeans(n_clusters = 3) #numero de clusters. Tentar encontrar no grafico o ponto de Elbow
kmeans.fit(base) #treina

centroides = kmeans.cluster_centers_
rotulos = kmeans.labels_

#define as cores do grafico
cores = ["g.", "r.", "b."]
for i in range(len(x)):
	#passa a coleção dos pontos, cada um com eixo x e y.
    plt.plot(base[i][0], base[i][1], cores[rotulos[i]], markersize = 15)
#imprime os centroides	
plt.scatter(centroides[:,0], centroides[:,1], marker = "x")
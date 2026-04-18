from minisom import MiniSom
import pandas as pd

base = pd.read_csv('wines.csv')
X = base.iloc[:,1:14].values
y = base.iloc[:,0].values

from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range = (0,1))
X = normalizador.fit_transform(X)

# x = quantas linhas o mapa vai ter.
# y = quantas colunas o mapa vai ter.
# input_len = quantos atributos.
# sigma = raio de alcance dos neurônios.
# learning rate = velocidade de atualização de aprendizado.
# random seed = inicializa os pesos para que sempre comece do mesmo e assim a aleatoriedade não interfere nos testes.
som = MiniSom(x = 8, y = 8, input_len = 13, sigma = 1.0, learning_rate = 0.5, random_seed = 2)
som.random_weights_init(X)
#num_iteration = épocas.
som.train_random(data = X, num_iteration = 100)

#som._weights
#som._activation_map
#quantas vezes cada neurônio foi selecionado como o Best Matching Unit. 
q = som.activation_response(X)

#Visualizar mapa

from pylab import pcolor, colorbar, plot
pcolor(som.distance_map().T) #.T = transposição (linha vira coluna e vice versa)
# MID - mean inter neuron distance
colorbar()

w = som.winner(X[2])
markers = ['o', 's', 'D']
color = ['r', 'g', 'b']
#y[y == 1] = 0
#y[y == 2] = 1
#y[y == 3] = 2

for i, x in enumerate(X):
    #print(i)
    #print(x)
    w = som.winner(x)
    #print(w)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = color[y[i]], markeredgewidth = 2)

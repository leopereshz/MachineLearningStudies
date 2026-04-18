import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


#divide as bases
(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()

#visualizar um registro (matriz) como imagem.
plt.imshow(X_treinamento[0], cmap = 'gray') #reduz a escala de cores para ser mais rápido.
#visualizar uma classe.
plt.title('Classe ' + str(y_treinamento[0]))



#necessario fazer uma transformação para que o tensorflow consiga fazer a leitura.
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0],28, 28, 1) #altura, largura e canal
#como vamos trabalhar em escala de cinzas, podemos informar 1 no canal.
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)

previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')


#normaliza os dados numa escala de 0 a 1 (float). Usa 255 porque é RGB.
previsores_treinamento /= 255
previsores_teste /= 255

#transforma a classe em variável dummy
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)


classificador = Sequential()
#a primeira camada é o operador de convulsão e o objetivo é gerar o mapa de características
#conv2D
	#32 - quantidade de mapas de características. Recomendável seguir 32, 64, 128, 256...
	#(3,3) - tamanho da matriz do detector de características
#input_shape - tamanho da imagem conforme definido lá em cima no reshape.
#activation = relu por padrão.
classificador.add(Conv2D(32, (3,3), input_shape=(28, 28, 1), activation = 'relu'))
#normaliza os resultados para ficar mais rápido.
classificador.add(BatchNormalization())
#pooling 1
classificador.add(MaxPooling2D(pool_size = (2,2))) #janela 2x2 para percorrer a matriz.

#mais uma camada de convulsão
classificador.add(Conv2D(32, (3,3), activation = 'relu'))
classificador.add(BatchNormalization())
#pooling 2
classificador.add(MaxPooling2D(pool_size = (2,2)))



#FLATTENING
classificador.add(Flatten()) #agora sim pode fazer o flatening



#GERA A REDE NEURAL DENSA

#aquela formula de neuronios na camada escondida não se aplica a redes convulcionais
classificador.add(Dense(units = 128, activation = 'relu')) 
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 10, activation = 'softmax'))

classificador.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento,batch_size = 128, epochs = 2,validation_data = (previsores_teste, classe_teste))

resultado = classificador.evaluate(previsores_teste, classe_teste)

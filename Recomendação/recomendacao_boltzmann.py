from rbm import RBM
import numpy as np

#num_visible = quantos nós visíveis (entradas) = numero de filmes.
#num_hidden = categorias de filmes (terror e comédia)
rbm = RBM(num_visible = 6, num_hidden = 2)

#6 usuários olharam 6 filmes de 2 categorias. 1- gostou, 0 - não gostou. Nesse exemplo não tem o conceito de não assistiu, então considera como 0.
base = np.array([[1,1,1,0,0,0],
                 [1,0,1,0,0,0],
                 [1,1,1,0,0,0],
                 [0,0,1,1,1,1],
                 [0,0,1,1,0,1],
                 [0,0,1,1,0,1]])

filmes = ["A bruxa", "Invocação do mal", "O chamado",
          "Se beber não case", "Gente grande", "American pie"]

rbm.train(base, max_epochs=5000) #5 mil é o máximo recomendado pelo criador do RBM.
rbm.weights

usuario1 = np.array([[1,1,0,1,0,0]])
usuario2 = np.array([[0,0,0,1,1,0]])

rbm.run_visible(usuario1) #indica qual neurônio foi ativado (comedia, terror)
rbm.run_visible(usuario2)

camada_escondida = np.array([[1,0]])
recomendacao = rbm.run_hidden(camada_escondida)

for i in range(len(usuario1[0])):
    #print(usuario1[0,i])
    if usuario2[0,i] == 0 and recomendacao[0,i] == 1:
        print(filmes[i])
    
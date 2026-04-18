import pandas as pd

base = pd.read_csv('0_risco_credito.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

#####################################################
#TRANSFORMAÇÕES
#####################################################                  
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])

#####################################################
#TREINAMENTO COM ÁRVORE DE DECISÃO
#####################################################
from sklearn.tree import DecisionTreeClassifier
classificador = DecisionTreeClassifier(criterion='entropy')
classificador.fit(previsores, classe)

#mostra qual a importância de cada atributo, conforme a ordem que está no dataset
print(classificador.feature_importances_)

#exportar a árvore de decisão para exibir graficamente
from sklearn.tree import export
export.export_graphviz(classificador,
                       out_file = 'arvore.dot',
                       feature_names = ['historia', 'divida', 'garantias', 'renda'],
                       class_names = ['alto', 'moderado', 'baixo'],
                       filled = True,
                       leaves_parallel=True)


#####################################################
#PREVER
#####################################################

# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
resultado = classificador.predict([[0,0,1,2], [3, 0, 0, 0]])

print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)



##############################
#BASE LINE CLASSIFIER
#GERA A PROBABILIDADE ATRAVÉS DE UM COUNT SIMPLES, PARA VOCÊ TESTAR SE VALE A PENA FAZER TODO O PROCESSAMENTO ACIMA
#pra achar o % tem que dividir o resultado pelo total de registros da amostra.
##############################
import collections
collections.counter(classe)
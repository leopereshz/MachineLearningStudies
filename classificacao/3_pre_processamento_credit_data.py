
import pandas as pd

#carrega os dados
base = pd.read_csv('0_credit-data.csv')

#estatísticas sobre os atributos
base.describe()

#####################################################
#TRATAR VALORES INCONSISTENTES
#####################################################

#localizar alguma informação através de um atributo
base.loc[base['age'] < 0]

#apagar coluna inteira. nomecoluna,colunainteira 1 ou 0, inplace = não retornar resultado, só executar.
base.drop('age',1,inplace=True) #parâmetros: coluna, apagar coluna inteira?, não retornar resposta?

#apagar somente os registros com problema
base.drop(base[base.age <0].index, inplace=True) #parâmetros: escolhe coluna, não retornar resposta?

#preencher os valores manualmente:
#ex1 - preencher os valores com a média
base.mean() #busca média de todos os campos
base['age'].mean() # buscar média de idade
base['age'][base.age>0].mean() #buscar a média de idade apenas das pessoas que tem idade maior que zero.
base.loc[base.age<0,'age'] = 40.92 #atualizar registros com idade negativa usando valor fixo.


#####################################################
#TRATAR VALORES FALTANTES
#####################################################

pd.isnull(base['age']) #roda todos os registros e valida um por um se a condição é true ou false.
base.loc[pd.isnull(base['age'])] #retorna apenas os registros que atendem ao filtro.


#CRIAR PREVISORES
#--------------------------------------
#atributo inicial do corte começam no zero, nesse exemplo ele ignorou o primeiro.
#atributo final do corte começa a contar a partir do 1. Vai entender.
previsores = base.iloc[:, 1:4].values
#pega somente uma coluna. Começa a contar a partir do zero.
classe = base.iloc[:, 4].values 

#tratar valores faltantes
#--------------------------------------
#importa biblioteca
from sklearn.preprocessing import Imputer
#cria o objeto e especifica os parâmetros de substituiçõa. Encontra os valores NaN utilizando a estratégia mean (média) e axis=0 para fazer atualização por colunas.
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
#quais as colunas quer que seja substituído.
imputer = imputer.fit(previsores[:, 0:3])
#substitui os valores propriamente ditos.
previsores[:,0:3] = imputer.transform(previsores[:,0:3])


#####################################################
#ESCALONAMENTO DE ATRIBUTOS
#utilizado principalmente no KNN.
#Escalonamento de todos os atributos na mesma escala evita que uma coluna como saldo seja considerada 
#mais importante do que outra com diferenças menores, como idade.

#Existem duas formas de fazer isso:
#Formula padronização: x= (x-media (x)) / desvio padrão(x)
#Fomrula normalização: x= (x-minimo(x)) / máximo(x)-mínimo(x)

#####################################################

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)


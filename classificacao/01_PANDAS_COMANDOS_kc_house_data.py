import pandas as pd
dataset = pd.read_csv('0_kc_house_data', sep=',') #sep = separador entre colunas.

dataset.head() #visualizar o valor da variável.

dataset.columns #imprimir todas as colunas.
dataset.count() #imprime quantas linhas tem o dataset.
pd.value_counts(dataset['bedrooms']) #imprime quantidade de imoveis agrupados pela quantidade de quartos.
dataset.loc[dataset['bedrooms']==3] #filtra apenas registros que tem determinado valor em uma coluna.
dataset.loc[(dataset['bedrooms']==3) & (dataset['bathrooms'] > 2)]

dataset.sort_values(by='price', ascending=False) #ordenar a saída por uma coluna, como ascendente ou descendente

dataset.describe() #estatísticas sobre a base.


dataset['size'] = (dataset['bedrooms']* 20) #definir o valor de uma coluna com base em outra. Essa fórmula vai aplicar para todas as linhas, cada um conforme suas colunas.

#definindo função
def categoriza(s):
    if s >= 80:
       return 'Big'
    elif s >= 60:
       return 'Medium'
    elif s >= 40:
       return 'Small'

#a coluna “cat_size” é criada a partir da aplicação da função “categoriza” a cada linha da coluna “size”.
dataset['cat_size'] = dataset['size'].apply(categoriza)


#excluir coluna. axis=1 = coluna. Inplace = aplica a alteração sobre a variável dataset, senão não tem efeito.
dataset.drop(['cat_size'], axis=1, inplace=True)

#excluir linha.
dataset.drop(dataset[dataset.bedrooms==0].index ,inplace=True)
dataset.drop(dataset[dataset.bedrooms>30].index ,inplace=True)

#conta registros em branco para cada coluna.
dataset.isnull().sum()

#remove linhas com valores faltantes.
dataset.dropna(inplace=True)
#remove linhas com valores faltantes em todas as colunas.
dataset.dropna(how='all', inplace=True)

#remove registros duplicados
dataset.drop_duplicates(subset="id", keep="first", inplace=True)

#preencher valores faltantes com um valor fixo.
dataset['bedrooms'].fillna(1, inplace=True)

#substituir valores errados
#o filtro é na coluna estado, pelos valores RP e PT, e se encontrar, atribuir na coluna estado o valor PR
dataset.loc[dataset["estado"].isin(['RP','PT']), 'estado'] = 'PR'

#preenche valores faltantes com a média da coluna.
dataset['floors'].fillna(dataset['floors'].mean(), inplace=True)

#identificar outliers em salário, considerando 2 desvios padrão
import statistics as sts
desv = sts.stdev(dataset['salario'])
dataset.loc[dataset['salario'] >= 2 * desv] #encontrar
#substituir pela mediana.
mediana = sts.median(dataset['salario'])
dataset.loc[dataset['salario'] >= 2 * desv, 'salario'] = mediana


#plotar gráfico simples no jupyter notebook:
%matplotlib notebook
dataset['price'].plot()
dataset.plot(x='bedrooms',y='price',kind='scatter', title='Bedrooms x Price',color='r')

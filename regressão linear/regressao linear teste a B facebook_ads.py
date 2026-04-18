#ex de analise de teste A|B.
#vendas com e sem facebook ads

from scipy.stats import linregress
from numpy import arange
import matplotlib.pyplot as plt

#0 = dias em que facebook ads estava desativado.
#1 = dias em que facebook ads estava ativado.
X = [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0]
#vendas em cada dia
Y = [12, 9, 13, 7, 10, 9, 12, 33, 31, 26, 37, 24, 27, 22, 14, 11, 8, 8, 10, 12, 11 ]

print(len(X),len(Y))

#cria um set para representar os dias.
x1 = arange(1,22)
print(len(x1))

plt.figure(figsize=(14,8))
plt.xlim(1,21)
plt.plot(x1, Y)
plt.show()

slope, intercept, r_value, p_value, std_err = linregress(X,Y)

#Intesecção = onde a linha do eixo Y encontra-se com o eixo X =0.
#ou seja, quando não tinha ads, o número de vendas era de média 10.
print(intercept)

#Inclinação (slope) - a cada unidade que aumenta a variável independente (X), a variável de resposta (y) sobe o valor da inclinação.
#ou seja, quanto cresceu desde que ativou o ads.
print(slope)


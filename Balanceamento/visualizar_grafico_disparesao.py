#EXEMPLO GERAL DE DESBALANCEADOS
from sklearn.datasets import make_classification
X, y = make_classification(
    n_classes=2, class_sep=1.5, weights=[0.99, 0.01],
    n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1,
    n_samples=10000, random_state=10
)
df = pd.DataFrame(X)
df['target'] = y
    
plot_2d_space(X, y, 'Exemplo de desbalanceamento')

#Esse tipo de situação é muito comum devido ao fato da classificação minoritária,
# representar normalmente cerca de 2–5% do todo.


target_count = df_train.target.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
target_count.plot(kind='bar', title='Count (target)',color = ['#1F77B4', '#FF7F0E']);

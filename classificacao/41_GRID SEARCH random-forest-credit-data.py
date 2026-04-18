import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

base = pd.read_csv('0_credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.cross_validation import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

###################
#TREINA FLORESTAS
###################

#cria coleções de parâmetros
grid_param = {
    'n_estimators': [10,25,50,75,100],
    'criterion': ['gini','entropy'],
    'bootstrap': [True,False]
}

classificador = RandomForestClassifier(random_state=0,max_depth=None)
gd_sr = GridSearchCV(estimator=classificador,param_grid=grid_param,scoring='accuracy',cv=5, n_jobs=-1)
gd_sr.fit(previsores_treinamento, classe_treinamento)

best_parameters = gd_sr.best_params_
best_result = gd_sr.best_score_



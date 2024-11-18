#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info" style="background-color: #DA70D6; border-color: #800080; color: #000000;">
#     <b>Introducción:</b> <a class="tocSkip"></a>
#     La compañía Megaline necesita un modelo que pueda analizar el comportamiento de sus usuarios para recomendar uno de sus nuevos planes: Smart o Ultra que tenga un humbral de exactitud de 0.75  
# </div>

# <div class="alert alert-block alert-info" style="background-color: #DDB0DD; border-color: #800080; color: #8B008B;">
#     <b></b> <a class="tocSkip"></a>
#     Primero importamos las librerías que podrían ser de utilidad para analizar los datos adecuadamente junto con el DataFrame. Partiendo de la premisa de que se quieren sustituir los planes anteriores, es decir voy a sustituir los planes Surf y Ultimate por Smart y Ultra, respectivamente, tomando como principal criterio, que el Paquete Smart ofrece 15GB y el Ultra incluye más de 15 GB. los usuarios, independientemente del plan que hayan utilizado anteriormente serán reasignados al nuevo plan que les corresponde, según el caso. Es necesario crear una columna nueva llamada 'is_ultra' donde 1 es que el usuario corresponde a ese plan, o 0 si no corresponde. Finalemte, dado que es un ejercicio de clasificación es pertinente utilizar métodos de clasificación como DecisionTreeClassifier, RandomForestClassifier y Regresión Logística con sus respectivos conjuntos de prueba, entrenamiento y validación. Al final la prueba de calidad para corroborarla con el conjunto de prueba. 
# </div>

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgbm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump


# In[2]:


users_behavior = pd.read_csv('/datasets/users_behavior_upd.csv')


# In[3]:


users_behavior.info()


# In[4]:


users_behavior.duplicated().sum()


# In[5]:


#users_behavior.drop_duplicates(inplace=True)
#users_behavior.duplicated().sum()


# <div class="alert alert-block alert-info" style="background-color: #DDB0DD; border-color: #800080; color: #8B008B;">
#     <b></b> <a class="tocSkip"></a>
#     El objetivo es entrenar un modelo con cualquiera de los modelos de clasificación como DecisionTreeClassifier, RandomForestClassifier o Regresión Lineal para clasificar a los usuarios de acuerdo a su actividad. En este caso, se probaron los tres para ver cuál se ajusta mejor al objetivo. Este Dataframe tiene homogeneidad en cuanto al formato de sus columnas; el tipo de datos de cada columna es el correcto. Los valores duplicados que existen se debe a que (como no existe un identificador único) es posible que haya filas con el mismo numero de llamadas, minutos, mensajes, megabytes usados y plan contratado, lo cual no interfiere con los resultados y no es necesario eliminarlos o tratarlos de alguna manera especial. Los conjuntos de prueba y validación comprenderán el 20 % de los datos.
# </div>

# In[6]:


users_behavior['is_ultra'] = users_behavior['is_ultimate']

users_behavior.head(20)


# In[7]:


users_behavior.loc[users_behavior['is_ultra'] > 15360, 'mb_used'] = 1
users_behavior.loc[users_behavior['is_ultra'] <= 15360, 'mb_used'] = 0

features = users_behavior.drop(['is_ultra','is_ultimate'], axis=1)
target = users_behavior['is_ultra']

print(features.head(10))
print(target.head(10))



# In[8]:


users_behavior_train, users_behavior_valid = train_test_split(users_behavior, test_size =0.40, random_state=666)
users_behavior_valid, users_behavior_test = train_test_split(users_behavior_valid, test_size=0.50, random_state=666)

features_train = users_behavior_train.drop(['is_ultra','is_ultimate'], axis=1)
target_train = users_behavior_train['is_ultra']
features_valid = users_behavior_valid.drop(['is_ultra','is_ultimate'], axis=1)
target_valid = users_behavior_valid['is_ultra']
features_test = users_behavior_test.drop(['is_ultra','is_ultimate'], axis=1)
target_test = users_behavior_test['is_ultra']

print(users_behavior_train.shape)
print(users_behavior_test.shape)
print(users_behavior_valid.shape)


# <div class="alert alert-block alert-info" style="background-color: #DDB0DD; border-color: #800080; color: #8B008B;">
#     <b></b> <a class="tocSkip"></a>
#     Es necesario dividir el dataframe en un conjunto de entrenamiento que tenga el 60% de los datos, otro conjunto de prueba también del 20% y un conjunto de validación del 20%, (1928 +643 + 643 = 3214)
# </div>

# <div class="alert alert-block alert-info" style="background-color: #DDB0DD; border-color: #800080; color: #8B008B;">
#     <b>MODELO 1. Árboles de decisión:</b> <a class="tocSkip"></a>
#     
# </div>

# In[9]:


best_score = 0
best_depth = 0
for depth in range(1, 21):
    model = DecisionTreeClassifier(random_state=666, max_depth=depth) 
    model.fit(features_train, target_train) 
    score = model.score(features_valid, target_valid) 
    if score > best_score:
        best_score = score
        best_depth = depth   
        
print("La exactitud del mejor modelo en el conjunto de validación (max_depth= {}): {}".format(best_depth, best_score))


# <div class="alert alert-block alert-info" style="background-color: #DDB0DD; border-color: #800080; color: #8B008B;">
#     <b>Resultados:</b> <a class="tocSkip"></a>
#     Para saber qué valor del hiperparámetro 'max_depth' es el más adecuado para entrenar el modelo, podemos crear un loop con un rango de valores, en este caso de profundidad. El valor que nos da una exactitud más cercana a 0.75 para el conjunto de prueba es <b>3</b>. 
# </div> 

# <div class="alert alert-block alert-info" style="background-color: #DDB0DD; border-color: #800080; color: #8B008B;">
#     <b>MODELO 2. Bosque aleatorio:</b> <a class="tocSkip"></a>
#     
# </div>

# In[21]:


best_error = 0
best_depth = 0
for est in range(1, 21):
    model_2 = RandomForestClassifier(random_state=666, n_estimators=est)
    model_2.fit(features_train, target_train) 
    predictions_valid = model_2.predict(features_valid) 
    error = mean_squared_error(target_valid, predictions_valid)**0.5
    score = model_2.score(features_valid, target_valid) 
    if score > best_score:
        best_score = score
        best_est = est
    if error < best_error: 
        best_error = error
        best_est = est
           
print("La exactitud del mejor modelo en el conjunto de validación (n_estimators = {}): {}" .format(best_est, best_score))


# <div class="alert alert-block alert-info" style="background-color: #DDB0DD; border-color: #800080; color: #8B008B;">
#     <b>Resultados:</b> <a class="tocSkip"></a>
#     Al probar un intervalo entre 1 y 20 El valor que nos da una exactitud más cercana a 0.75 para el conjunto de prueba es 14 del hiperparámetro 'n_estimators' el cual nos da 0.74, lo cual se acerca bastante al valor deseado.
# </div>

# <div class="alert alert-block alert-info" style="background-color: #DDB0DD; border-color: #800080; color: #8B008B;">
#     <b>MODELO 3. Regresión Logística:</b> <a class="tocSkip"></a>
#     
# </div>

# In[16]:


data_ohe = pd.get_dummies(users_behavior, drop_first=True)
features_ohe = data_ohe.drop(['is_ultra','is_ultimate'], axis=1)
target_ohe = data_ohe['is_ultra']

features_ohe_train, features_ohe_valid, target_ohe_train, target_ohe_valid = train_test_split(features_ohe, target_ohe, test_size=0.25, random_state=12345)

model_3 = LogisticRegression(random_state=12345, solver='liblinear')
model_3.fit(features_ohe_train, target_ohe_train)

score_train = model.score(features_ohe_train, target_ohe_train)
score_valid = model.score(features_ohe_valid, target_ohe_valid)

print("Exactitud del modelo de regresión logística en el conjunto de entrenamiento:", score_train)
print("Exactitud del modelo de regresión logística en el conjunto de validación:", score_valid)




# <div class="alert alert-block alert-info" style="background-color: #DDB0DD; border-color: #800080; color: #8B008B;">
#     <b>Resultados:</b> <a class="tocSkip"></a>
#     Para este modelo ustilicé primero el método get_dummies para obtener una clasificación binaria. El modelo da una exactitud por arriba del 0.75 tanto para el conjunto de validacion como entrenamiento (0.86), el cual es un valor mucho mejor que el valor deseado.
# </div>

# <div class="alert alert-block alert-info" style="background-color: #DDB0DD; border-color: #800080; color: #8B008B;">
#     <b>Aplicando el mejor modelo al conjunto de prueba:</b> <a class="tocSkip"></a>
#     
# </div>

# In[17]:


model_3.fit(features_test, target_test)
score_test = model_3.score(features_test, target_test)

print('Exactitud del modelo 3 aplicado al conjunto de prueba:', score_test)


# In[ ]:


joblib.dump(model_3, 'model_3.joblib')


# <div class="alert alert-block alert-info" style="background-color: #DDB0DD; border-color: #800080; color: #8B008B;">
#     <b>Conclusión:</b> <a class="tocSkip"></a>
#     El modelo que tuvo una exactitud mayor en el conjunto de validación fue La Regresión Logística con el método get_dummies,  con una exactitud muy por arriba del 0.75, lo cual podría ser una buena opción para realizar la reclasififcación de usuarios. los modelos basados en árboles también tuvieron exactitudes muy cercanas al 0.75 en el conjunto de validación. Al final, decidí usar La Regresión Logística para aplicar el modelo al conjunto de prueba ya que la exactitud tanto en el conjunto de prueba ocmo en el conjunto de validacion fue el mismo valor, lo cual indica que no hay subajuste ni sobreajuste.
# </div>

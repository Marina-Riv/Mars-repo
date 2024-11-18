#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info" style="background-color: #00CED1; border-color: #4682B4; color: #FFFFFF;">
#     <b>Introducción:</b> <a class="tocSkip"></a>
#     Recientemente ha habido una disminución en los usuarios de BetaBank. Estratégicamente es mejor salvar los clientes existentes que atraer nuevos. Es por ello, que basado en los datos pasados de comportamiento de clientes y contratos de terminación, es necesario crear un modelo para predecir qué clientes es probable que terminen contrato con el banco.  
# </div>

# <div class="alert alert-block alert-info" style="background-color: #AFEEEE; border-color: #4682B4; color: #4682B4;">
#     <b>Procedimiento:</b> <a class="tocSkip"></a>
#       Primero es necesario importar las librerías para preparar los datos y para crear un modelo que se ajuste a las necesidades del cliente al momento de entrenarlo. En este proceso, se prueban diferentes modelos de clasificación para ver cual da un mejor resultado. Después se carga el dataset de interés, en este caso es la información histórica del comportamiento de los clientes en cuanto al servicio de BetaBank. En este paso se revisa que los datos tengan el formato adecuado, si hay duplicados o valores ausentes, para poder aplicar los modelos adecuadamente. Posteriormente, se divide en conjuntos de entrenamiento, validación y prueba para poder probar los modelos. Se hacen pruebas para ajustar algunos hiperparámetros, se revisa el equilibrio  el dataset. De acuerdo con el tipo de datos de tengamos (categoricos o numéricos) se usará un "ordinal encoder" (para convertir todos los datos categoricos en numéricos de todas las columnas que lo requieran) y una estandarización de datos si estos están en diferentes escalas, finalmente, en caso de encontrar un desequilibrio de clases se aplicará un método de sobremuesttreo o submuestreo para equilibrarlas.
# </div>

# In[1]:


import math as math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle


# In[2]:


churn = pd.read_csv('/datasets/Churn.csv')
churn.info()


# In[3]:


churn.head(40)


# In[4]:


print(churn['Tenure'].mean())
print(churn['Tenure'].median())


# In[5]:


columns_to_replace = ['Tenure']
for col in columns_to_replace:
    churn['Tenure'].fillna('-1', inplace=True)
    
print(churn['Tenure'].isna().sum())

def roundup (salary):
    return math.ceil (salary)
churn['Balance'] = churn['Balance'].apply(roundup)
churn['Balance'] = churn['Balance'].astype(int)
churn['EstimatedSalary'] = churn['EstimatedSalary'].apply(roundup)
churn['EstimatedSalary'] = churn['EstimatedSalary'].astype(int)
churn['Tenure'] = churn['Tenure'].astype(int)
encoder = OrdinalEncoder()

churn = pd.DataFrame(encoder.fit_transform(churn), columns=churn.columns)


# In[6]:


churn.sample(20)


# In[7]:


churn['Tenure'].value_counts(dropna=False).sort_index()


# In[8]:


churn.isna().sum()


# In[9]:


new_col_names=[]
for old_name in churn.columns:
    name_lowered=old_name.lower()
    new_col_names.append(name_lowered)
churn.columns=new_col_names

column_new={'rownumber': 'row_number', 'customerid': 'customer_id', 'surname':'surname', 'creditscore': 'credit_score', 'geography': 'geography', 'gender': 'gender', 'age': 'age', 'tenure' : 'tenure', 'balance': 'balance', 'numofproducts': 'num_of_products', 'hascrcard': 'has_crcard',
       'isactivemember': 'is_active_member', 'estimatedsalary': 'estimated_salary', 'exited': 'exited'}
    
churn=churn.rename(columns=column_new)
print(churn.columns)


# In[10]:


churn.duplicated().sum()


# <div class="alert alert-block alert-info" style="background-color: #AFEEEE; border-color: #4682B4; color: #4682B4;">
#     <b>Procesamiento:</b> <a class="tocSkip"></a>
#       Al revisar el DataFrame encontré tres diferentes tipos de datos: enteros, continuos y categóricos. los continuos los convertí en enteros, los categóricos a numéricos con el método OrdinalEncoder, estos cambios los almacené en una nueva variable llamada churn_ordinal. Posteriormente, homogeneicé los nombres de las columnas para que todos estuvieran en minúsculas separados por un guión bajo. La columna 'Tenure' tenía valores ausentes que sustituí por -1 para tratarlos como una categoría aparte. Esta codificación de datos es apropiada para realizar un modelo basado en árboles ya que aunque el valor numerico de una categoría sea alto no se le atribuye mayor valor, es decir, trata a cada categoría por igual, por lo que el primer modelo utilizado es 'DecisionTreeClassifier' con los respectivos conjuntos de entrenamiento y validación. 
# </div>

# In[11]:


churn['churn'] = churn['exited']

churn.sample(40)


# In[12]:


features = churn.drop(['churn', 'exited', 'tenure'], axis=1)
target = churn['churn']

print(features.head(10))
print(target.head(10))


# In[13]:


pd.options.mode.chained_assignment = None

features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.40, random_state=2684
)
features_valid, features_test, target_valid, target_test = train_test_split(
    features_valid, target_valid, test_size=0.50, random_state=2684
)
numeric = ['credit_score', 'age', 'balance', 'estimated_salary']

scaler = StandardScaler()
scaler.fit(features_train[numeric])

features_train[numeric] = scaler.transform(features_train[numeric])

features_valid[numeric] = scaler.transform(features_valid[numeric])

features_test[numeric] = scaler.transform(features_test[numeric])

print(features_train.shape)
print(features_test.shape)
print(features_valid.shape)
print(features_train.head(5))


# <div class="alert alert-block alert-info" style="background-color: #AFEEEE; border-color: #4682B4; color: #4682B4;">
#     <b>Nota:</b> <a class="tocSkip"></a> 
#         También fue necesario estandarizar los valores de las columnas con el metodo StandardScaler() ya que algunas varían entre sí hasta dos órdenes de magnitud.
# </div>

# <div class="alert alert-block alert-info" style="background-color: #AFEEEE; border-color: #4682B4; color: #4682B4;">
#     <b>Modelos</b> <a class="tocSkip"></a>
#     
# </div>

# <div class="alert alert-block alert-info" style="background-color: #AFEEEE; border-color: #4682B4; color: #4682B4;">
#     <b>Regresión Logística:</b> <a class="tocSkip"></a>
#     
# </div>

# In[14]:


model = LogisticRegression(random_state=2684, solver='liblinear')
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)
accuracy_valid = accuracy_score(target_valid, predicted_valid)
predicted_valid_series = pd.Series(model.predict(features_valid))
class_frequency = predicted_valid_series.value_counts(normalize = True)
target_pred_constant = pd.Series(0, index=target_valid.index)

print("Exactitud del modelo:", accuracy_valid)
print("Porcentaje de respuestas correctas:", accuracy_score(target_valid, target_pred_constant),"%")
print("Frecuencia de clases:", class_frequency)
print("F1:", f1_score(target_valid, predicted_valid))


# <div class="alert alert-block alert-info" style="background-color: #AFEEEE; border-color: #4682B4; color: #4682B4;">
#     <b>Resultados:</b> <a class="tocSkip"></a> 
#         La exactitud del modelo es de 81%, el porcentaje de espuestas correctas es 79%, el valor F1 del modelo es muy bajo (0.28), hay una mayor cantidad de usuarios que no estan contentos con el servicio de BetaBank, es decir hay un enorme desequilibrio de clases lo cual se muestra gráficamente en la siguiente matriz de confusión:
# </div>

# <div class="alert alert-block alert-info" style="background-color: #AFEEEE; border-color: #4682B4; color: #4682B4;">
#     <b>Matriz de confusión:</b> <a class="tocSkip"></a> 
# </div>

# In[15]:


class_frequency.plot(kind='bar')
print("Matriz de confusión:", (confusion_matrix(target_valid, predicted_valid)))
print("Valor del Recall_Score es:", recall_score(target_valid, predicted_valid))
print("Valor de la precisión es:", precision_score(target_valid, predicted_valid))


# <div class="alert alert-block alert-info" style="background-color: #AFEEEE; border-color: #4682B4; color: #4682B4;">
#     <b>Equilibrando clases con sobremuestreo:</b> <a class="tocSkip"></a> 
# </div>

# In[25]:


repeat = 4800
def upsample(features, target, repeat):
    features_zeros = features_train[target_train == 0]
    features_ones = features_train[target_train == 1]
    target_zeros = target_train[target_train == 0]
    target_ones = target_train[target_train == 1]
    
    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat )
    
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat )

    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=2684
    )
    
    return features_upsampled, target_upsampled

features_upsampled, target_upsampled = upsample(
    features_train, target_train, 4800
)


# <div class="alert alert-block alert-info" style="background-color: #AFEEEE; border-color: #4682B4; color: #4682B4;">
#     <b>Valor F1 del modelo de Regresión Logística con sobremuestreo:</b> <a class="tocSkip"></a>
#     
# </div>

# In[27]:


model_up = LogisticRegression(random_state=2684, solver='liblinear')
model_up.fit(features_upsampled, target_upsampled)
predicted_valid_up = model.predict(features_valid)

print('F1:', f1_score(target_valid, predicted_valid))

features_zeros = features_train[target_train == 0]
features_ones = features_train[target_train == 1]
target_zeros = target_train[target_train == 0]
target_ones = target_train[target_train == 1]

print(features_zeros.shape)
print(features_ones.shape)
print(target_zeros.shape)
print(target_ones.shape)


# <div class="alert alert-block alert-info" style="background-color: #AFEEEE; border-color: #4682B4; color: #4682B4;">
#     <b>Equilibrando clases con submuestreo:</b> <a class="tocSkip"></a> 
# </div>

# In[18]:


def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)]
        + [features_ones]
    )
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)]
        + [target_ones]
    )

    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345
    )

    return features_downsampled, target_downsampled


features_downsampled, target_downsampled = downsample(
    features_train, target_train, 0.5
)


# <div class="alert alert-block alert-info" style="background-color: #AFEEEE; border-color: #4682B4; color: #4682B4;">
#     <b>Valor F1 del modelo de Regresión Logística con submuestreo:</b> <a class="tocSkip"></a>
#     
# </div>

# In[19]:


model_down = LogisticRegression(random_state=2684, solver='liblinear')
model_down.fit(features_downsampled, target_downsampled)
predicted_valid_down = model_down.predict(features_valid)
print('F1:', f1_score(target_valid, predicted_valid))


# <div class="alert alert-block alert-info" style="background-color: #AFEEEE; border-color: #4682B4; color: #4682B4;">
#     <b>Valor AUC-ROC del model de Regresíon Logística (con sobremuestreo):</b> <a class="tocSkip"></a>
#     
# </div>

# In[20]:


predictions_valid = model_up.predict(features_valid)

probabilities_valid = model_up.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

print("F1 del conjunto de prueba:", f1_score(target_test, predictions_valid))
fpr, tpr, thresholds = roc_curve(target_valid, probabilities_one_valid)

plt.figure()

plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Tasa de verdaderos positivos")
plt.title("Curva ROC")

auc_roc = roc_auc_score (target_valid, probabilities_one_valid)
print("Valor del área bajo la curva ROC:", auc_roc)


# <div class="alert alert-block alert-info" style="background-color: #AFEEEE; border-color: #4682B4; color: #4682B4;">
#     <b>Entrenamiento de otros modelos:</b> <a class="tocSkip"></a>
#     
# </div>

# <div class="alert alert-block alert-info" style="background-color: #AFEEEE; border-color: #4682B4; color: #4682B4;">
#     <b>DecisionTreeClassifier:</b> <a class="tocSkip"></a>
#     
# </div>

# In[21]:


for depth in range(1, 21):
    model_2 = DecisionTreeClassifier(random_state=12345, criterion='gini', max_depth=depth)
    model_2.fit(features_train, target_train)
    predictions_valid = model_2.predict(features_valid)
    print("max_depth =", depth, ": ", end='')
    print(accuracy_score(target_valid, predictions_valid))


# In[22]:


model_2 = DecisionTreeClassifier(random_state=2684, criterion='gini', class_weight='balanced', max_depth=5)
model_2.fit(features_train, target_train)
predictions_valid = model_2.predict(features_valid)
predictions_test = model_2.predict(features_valid)
print("Exactitud del modelo modificado:", accuracy_score(target_valid, predictions_valid))
print("F1 del conjunto de prueba:", f1_score(target_valid, predictions_valid))


# <div class="alert alert-block alert-info" style="background-color: #AFEEEE; border-color: #4682B4; color: #4682B4;">
#     <b>Resultados del modelo DecisionTreeClassifier:</b> <a class="tocSkip"></a>
#     la Exactitud del modelo del 80% muy similar al modelo de Regresión logística, sin embargo el valor F1 es de 0.59. 
# </div>

# <div class="alert alert-block alert-info" style="background-color: #AFEEEE; border-color: #4682B4; color: #4682B4;">
#     <b>Random Forest Classifier:</b> <a class="tocSkip"></a>
#     
# </div>

# In[23]:


best_score = 0
best_est = 0
for est in range(1, 101): 
    model_3 = RandomForestClassifier(random_state=2684, n_estimators=est) 
    model_3.fit(features_train, target_train) 
    score = model_3.score(features_valid, target_valid) 
    if score > best_score:
        best_score = score 
        best_est = est
print("La exactitud del mejor modelo en el conjunto de validación (n_estimators = {}): {}".format(best_est, best_score))


# In[24]:


model_3 = RandomForestClassifier(random_state=2684, n_estimators=78, class_weight='balanced') 
model_3.fit(features_train, target_train)
predictions_valid_3 = model_3.predict(features_valid)
print("F1 del conjunto de validación:", f1_score(target_valid, predictions_valid_3))


# <div class="alert alert-block alert-info" style="background-color: #AFEEEE; border-color: #4682B4; color: #4682B4;">
#     <b>Resultados del modelo RandomForestClassifier:</b> <a class="tocSkip"></a>
#     la Exactitud del modelo del 86% mayor a los modelos Regresión logística y DecisionTreeClassifier, en este caso el valor F1 tampoco es suficiente para alcanzar 0.59. 
# </div>

# <div class="alert alert-block alert-info" style="background-color: #AFEEEE; border-color: #4682B4; color: #4682B4;">
#     <b>Conclusión:</b> <a class="tocSkip"></a>
#     El modelo que tuvo un mayor exactitud fue RandomForestClassifier con 86%, sin embargo el modelo con mayor valor de F1 fue el DecisionTreeClassifier con 0.59 a pesar de que al Regresión Logpistica tuvo una valor  de AUC-ROC aceptable superior a 0.7
#     
# </div>

# In[ ]:





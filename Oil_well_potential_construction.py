#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info" style="background-color: #F4A460; border-color: #FFFFFF; color: #8B008B;">
#     <b>Introducción:</b> <a class="tocSkip"></a>
#       La empresa OilyGiant busca abrir 200 pozos nuevos de petróleo, con la ayuda de un modelo y datos sobre muestras de producto de tres regiones se pretende elegir el lugar con mayor margen de beneficio y menor riesgo de pérdidas. 
# </div>

# <div class="alert alert-block alert-info" style="background-color: #8B008B; border-color: #FFFFFF; color: #F4A460;">
#     <b>Procedimiento y Objetivo:</b> <a class="tocSkip"></a>
#       El principal objetivo es encontrar los mejores lugares donde abrir 200 pozos nuevos de petróleo basado en datos sobre muestras de pozos petrolíferos de tres regiones. Se creó un modelo de machine learning que ayuda a elegir el lugar con mayor margen de beneficio y al mismo tiempo evaluando los riesgos de pérdidas.
# </div>

# In[1]:


import math as math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import RandomState
from scipy import stats as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split


# In[2]:


region_0 = pd.read_csv('/datasets/geo_data_0.csv')
region_1= pd.read_csv('/datasets/geo_data_1.csv')
region_2 = pd.read_csv('/datasets/geo_data_2.csv')


# In[3]:


region_0.info()


# In[4]:


region_1.info()


# In[5]:


region_2.info()


# In[6]:


region_0.isna().sum()


# In[7]:


region_1.isna().sum()


# In[8]:


region_2.isna().sum()


# In[9]:


region_0.duplicated().sum()


# In[10]:


region_1.duplicated().sum()


# In[11]:


region_2.duplicated().sum()


# In[12]:


region_0.head(20)


# In[13]:


region_1.head(20)


# In[14]:


region_2.head(20)


# In[15]:


region_0 ['predicted_product'] = region_0['product']
region_1 ['predicted_product'] = region_1['product']
region_2 ['predicted_product'] = region_2['product']


# *No hay valores ausentes, no hay formatos extraños ni tipos de datos dificiles de trabajar y no hay valores duplicados en ninguno de los sitios.

# <div class="alert alert-block alert-info" style="background-color: #F4A460; border-color: #FFFFFF; color: #8B008B;">
#     <b>Analisis de las características f0,f1, f2 y product de las tres regiones:</b> <a class="tocSkip"></a> 
# </div>

# <div class="alert alert-block alert-info" style="background-color: #8B008B; border-color: #FFFFFF; color: #F4A460;">
#     <b>region_0</b> <a class="tocSkip"></a>
# </div>

# In[16]:


#construcción de un histograma
columns = ['f0', 'f1', 'f2', 'product']

fig, axes= plt.subplots(1, len(columns), figsize=(15,4))
for ax, column in zip(axes, columns):
        region_0[column].hist(ax=ax, alpha=0.7, bins=50, color='m')

fig.suptitle("Distribución datos f0, f1, f2 (region_0)", y=0.96)
axes[0].set_title('"f0"')
axes[1].set_title('"f1"')
axes[2].set_title('"f2"')
axes[3].set_title('"product"')

plt.tight_layout()
plt.show()


# In[17]:


correlation_matrix = region_0.corr()
correlation_with_product = correlation_matrix['product'][['f0', 'f1', 'f2']]
print(correlation_with_product)


# *Existe una correlación positiva entre el producto y la característica f2

# <div class="alert alert-block alert-info" style="background-color: #8B008B; border-color: #FFFFFF; color: #F4A460;">
#     <b>region_1</b> <a class="tocSkip"></a>
# </div>

# In[18]:


columns = ['f0', 'f1', 'f2', 'product']

fig, axes= plt.subplots(1, len(columns), figsize=(15,4))
for ax, column in zip(axes, columns):
        region_1[column].hist(ax=ax, alpha=0.7, bins=50)

fig.suptitle("Distribución datos f0, f1, f2 (region_1)", y=0.96)
axes[0].set_title('"f0"')
axes[1].set_title('"f1"')
axes[2].set_title('"f2"')
axes[3].set_title('"product"')

plt.tight_layout()
plt.show()


# In[19]:


correlation_matrix = region_1.corr()
correlation_with_product = correlation_matrix['product'][['f0', 'f1', 'f2']]
print(correlation_with_product)


# *Existe una alta correlación positiva entre el producto y la característica f2

# <div class="alert alert-block alert-info" style="background-color: #8B008B; border-color: #FFFFFF; color: #F4A460;">
#     <b>region_2</b> <a class="tocSkip"></a>
# </div>

# In[20]:


columns = ['f0', 'f1', 'f2', 'product']

fig, axes= plt.subplots(1, len(columns), figsize=(15,4))
for ax, column in zip(axes, columns):
        region_2[column].hist(ax=ax, alpha=0.7, bins=50, color='g')

fig.suptitle("Distribución datos f0, f1, f2 (region_2)", y=0.96)
axes[0].set_title('"f0"')
axes[1].set_title('"f1"')
axes[2].set_title('"f2"')
axes[3].set_title('"product"')

plt.tight_layout()
plt.show()


# In[21]:


correlation_matrix = region_2.corr()
correlation_with_product = correlation_matrix['product'][['f0', 'f1', 'f2']]
print(correlation_with_product)


# *Existe una correlación positiva entre el producto y la característica f2

# <div class="alert alert-block alert-info" style="background-color: #F4A460; border-color: #FFFFFF; color: #8B008B;">
#     <b>Función de entenamiento aplicable a las regiones:</b> <a class="tocSkip"></a> 
# </div>

# In[22]:


def training (data, random_state=123):
    features = data.drop(columns=['id', 'product', 'predicted_product'], axis=1)
    target = data['predicted_product']
    features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=123)
    model = LinearRegression()
    model.fit(features_train, target_train)
    predictions_valid = model.predict(features_valid)
    rmse = np.sqrt(mean_squared_error(target_valid, predictions_valid))
    print('RMSE:', rmse)
    predictions = model.predict(features) # Aquí tengo duda en si se saca el promedio y se guardan las predicciones de todo el dataframe 
    #o del conjunto de validación porque al guardar el conjunto de validación al momento de obtener los 200 pozos con mayores reservas 
    # únicamente podemos utilizar los pozos del conjunto de validación y no los que realmente tienen mayores reservas de todo el DF.
    data['predicted_product'] = predictions
    mean_predict = predictions.mean()
    mean_predict_valid = predictions_valid.mean()
    print('Volumen total predicho promedio de reservas:', mean_predict)
    print('Volumen predicho promedio de reservas del conjunto de validación:', mean_predict_valid)
    return data


# <div class="alert alert-block alert-info" style="background-color: #8B008B; border-color: #FFFFFF; color: #F4A460;">
#     <b>Entrenamiento de modelo para la region_0:</b> <a class="tocSkip"></a>
#       
# </div>

# In[23]:


predicted_region_0 = training(region_0)
print(predicted_region_0)


# <div class="alert alert-block alert-info" style="background-color: #8B008B; border-color: #FFFFFF; color: #F4A460;">
#     <b>Entrenamiento de modelo para la region_1:</b> <a class="tocSkip"></a>
#       
# </div>

# In[24]:


predicted_region_1 = training(region_1)
print(predicted_region_1)


# <div class="alert alert-block alert-info" style="background-color: #8B008B; border-color: #FFFFFF; color: #F4A460;">
#     <b>Entrenamiento de modelo para la region_2:</b> <a class="tocSkip"></a>
#       
# </div>

# In[25]:


predicted_region_2 = training(region_2)
print(predicted_region_2)


# *La región que tuvo menor error es la region_1 (RMSE: 0.89). Sin embargo, al enfocarnos en la variable dependiente ('predicted_product') se observa que la region_2 tiene una predicción mayor de reservas (95.09 (miles de barriles)) comparado con las demás regiones, seguido de la región_0 con 92.54 (miles de barriles), finalmente, la region_1 con 69.28 (miles de barriles). 

# <div class="alert alert-block alert-info" style="background-color: #F4A460; border-color: #FFFFFF; color: #8B008B;">
#     <b>Cálculo de ganancias:</b> <a class="tocSkip"></a>
#      
# </div>

# In[26]:


total_invest = 100000000
well_num = 200
mean_prod_per_well = 111.1
bare_profit_per_well = 4500
profit_per_well = total_invest/well_num
print('Ganancia por pozo:', profit_per_well)
prod_per_well = profit_per_well/mean_prod_per_well
print('Producción necesaria por pozo (en dls):', prod_per_well)


# In[27]:


def profit (data, total_invest = 100000000, well_num = 200, mean_prod_per_well = 111.1, bare_profit_per_well = 4500):
    profit_per_well = total_invest/well_num
    
    prod_per_well = profit_per_well / mean_prod_per_well
    
    real_profit = data['predicted_product'].sum()*bare_profit_per_well-total_invest
    #data['real_profit'] = data['predicted_product'].sum()*bare_profit_per_well-total_invest
    return real_profit
    


# In[ ]:





# <div class="alert alert-block alert-info" style="background-color: #8B008B; border-color: #FFFFFF; color: #F4A460;">
#     <b>ganancias potenciales de los 200 pozos principales para la region_0</b> <a class="tocSkip"></a>
#       
# </div>

# In[28]:


top_200_wells_0 = predicted_region_0.nlargest(200, 'predicted_product').reset_index()
print(top_200_wells_0)


# In[29]:


print(profit(top_200_wells_0))


# In[30]:


potencial_profit_0 = top_200_wells_0['predicted_product'].sum() * bare_profit_per_well - total_invest
print("La ganancia potencial de la region_0 de los 200 mejores pozos:", potencial_profit_0)


# <div class="alert alert-block alert-info" style="background-color: #8B008B; border-color: #FFFFFF; color: #F4A460;">
#     <b>ganancias potenciales de los 200 pozos principales para la region_1</b> <a class="tocSkip"></a>
#       
# </div>

# In[31]:


top_200_wells_1 = predicted_region_1.nlargest(200, 'predicted_product').reset_index()
print(top_200_wells_1)


# In[32]:


profit(top_200_wells_1)


# In[33]:


potencial_profit_1 = top_200_wells_1['predicted_product'].sum() * bare_profit_per_well - total_invest
print("La ganancia potencial de la region_1 de los 200 mejores pozos:", potencial_profit_1)


# <div class="alert alert-block alert-info" style="background-color: #8B008B; border-color: #FFFFFF; color: #F4A460;">
#     <b>ganancias potenciales de los 200 pozos principales para la region_2</b> <a class="tocSkip"></a>
#       
# </div>

# In[34]:


top_200_wells_2 = predicted_region_2.nlargest(200, 'predicted_product').reset_index()
print(top_200_wells_2)


# In[35]:


profit(top_200_wells_2)


# In[36]:


potencial_profit_2 = top_200_wells_2['predicted_product'].sum() * bare_profit_per_well - total_invest
print("La ganancia potencial de la region_2 de los 200 mejores pozos:", potencial_profit_2)


# *Comparando el beneficio del producto predicho cada región, tomando en cuenta los 200 pozos con mayores reservas, la región con mayor beneficio es la region_0 (con 46,785,836.15 dls), seguido de la region_2 (con 41,386,421.33 dls). 

# <div class="alert alert-block alert-info" style="background-color: #8B008B; border-color: #FFFFFF; color: #F4A460;">
#     <b>Bootstrapping y análisis de riesgo:</b> <a class="tocSkip"></a>
#       
# </div>

# In[37]:


state=np.random.RandomState(123)

def bootstrapping(data, num_muestras=1000):
    profit_per_sample = [] 
    
    for _ in range(num_muestras):
        subsample = data.sample(n=500, replace=True, random_state=state)
        profit_per_sample.append(profit(subsample))
    
    real_profit = pd.Series(profit_per_sample)
    return real_profit

print("Las ganancias promedio para la region_0 de acuerdo con el método de bootstrapping es:", bootstrapping(top_200_wells_0).mean())
print("Las ganancias promedio para la region_1 de acuerdo con el método de bootstrapping es:", bootstrapping(top_200_wells_1).mean())
print("Las ganancias promedio para la region_2 de acuerdo con el método de bootstrapping es:", bootstrapping(top_200_wells_2).mean())
print()
print("Las ganancias promedio para datos reales de la region_0 de acuerdo con el método de bootstrapping es:", bootstrapping(predicted_region_0).mean())
print("Las ganancias promedio para datos reales de la region_1 de acuerdo con el método de bootstrapping es:", bootstrapping(predicted_region_1).mean())
print("Las ganancias promedio para datos reales de la region_2 de acuerdo con el método de bootstrapping es:", bootstrapping(predicted_region_2).mean())


# *Comparando un bootstrapping utilizando los mejores pozos de cada región contra un bootstrapping utilizando pozos de cada región aleatoriamente para 1000 iteraciones y muestras de 500 tenemos que para el bootstrapping de los mejores pozos la region con mayores ganancias es la region_0, aunque no por mucho comparado con las demás regiones. Por otro lado, el bootstrapping aleatorio nos dice que la región_1 tendrá mejores ganancias (54,888,840.86dls). Esto discrepa del resultado anterior con la función con calculos aritméticos que mostraba a la region_2 como la que tenia mejores ganancias. 

# In[38]:


bootstrap_region_0 = bootstrapping(predicted_region_0)
bootstrap_region_1 = bootstrapping(predicted_region_1)
bootstrap_region_2 = bootstrapping(predicted_region_2)


# In[39]:


results = {
    'Region 0': (bootstrap_region_0.mean(), bootstrap_region_0.std()),
    'Region 1': (bootstrap_region_1.mean(), bootstrap_region_1.std()),
    'Region 2': (bootstrap_region_2.mean(), bootstrap_region_2.std())
}


for region, (mean_profit, std_profit) in results.items():
    print(f'Ganancias promedio para {region}: {mean_profit:.2f} ± {std_profit:.2f}')


# In[40]:


def calculate_statistics(bootstrap_results):
    mean_profit = bootstrap_results.mean()
    confidence_interval = bootstrap_results.quantile(0.975) - bootstrap_results.quantile(0.025)
    loss_probability = (bootstrap_results < 0).mean() * 100  
    return mean_profit, confidence_interval, loss_probability

print("El beneficio promedio, intervalo de confianza y probabilidades de pérdida de la Región 0 es:", calculate_statistics(bootstrap_region_0))
print("El beneficio promedio, intervalo de confianza y probabilidades de pérdida de la Región 1 es:", calculate_statistics(bootstrap_region_1))
print("El beneficio promedio, intervalo de confianza y probabilidades de pérdida de la Región 2 es:", calculate_statistics(bootstrap_region_2))
           


# * Ninguna región tuvo riesgo de perdidas, únicamente beneficios y la region con mayor beneficios fue a region 1.
# 

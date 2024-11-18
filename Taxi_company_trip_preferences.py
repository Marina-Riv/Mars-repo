#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info" style="background-color: #DA70D6; border-color: #800080; color: #8B008B;">
#     <b>Introduccion:</b> <a class="tocSkip"></a>
#     Con la informacion de los viajes de diversas compañías de taxis la empresa Zuber busca identificar los destinos más populares en Chicago y visualizar de qué forma el clima impacta los viajes para establecer un sistema de transporte.  
# </div>

# In[1]:


from scipy import stats as st
from scipy.stats import ttest_ind
from scipy.stats import levene
from matplotlib import pyplot as plt
import math as mt
import numpy as np
import pandas as pd 
import seaborn as sns 
import scipy as sp


# In[2]:


sql_result_01 = pd.read_csv('/datasets/project_sql_result_01.csv')


# In[3]:


print(sql_result_01.info())


# In[4]:


sql_result_01.isna().sum()


# In[5]:


sql_result_01.duplicated().sum()


# In[6]:


sql_result_04 = pd.read_csv('/datasets/project_sql_result_04.csv')


# In[7]:


print(sql_result_04.info())


# In[8]:


sql_result_04.isna().sum()


# In[9]:


sql_result_04.duplicated().sum()


# <div class="alert alert-block alert-info" style="background-color: #DDB0DD; border-color: #800080; color: #000000;">
#     <b>♣</b> <a class="tocSkip"></a>
#     10 principales barrios desitno:
# </div>

# In[10]:


top_10_destinations = sql_result_04.head(10)


# In[11]:


sns.barplot(data=top_10_destinations, x='dropoff_location_name', y='average_trips')
plt.ylabel('Viajes promedio')
plt.xlabel('Destinos')
plt.xticks(rotation=90)
plt.title('Destinos populares de acuerdo al número promedio de viajes en el mes de Noviembre')


# <div class="alert alert-block alert-info" style="background-color: #DDB0DD; border-color: #800080; color: #000000;">
#     <b></b> <a class="tocSkip"></a>
#     El destino final más recurrente es Loop (en color azul) con más de 10000 viajes para los días 15 y 16 de Noviembre. El segundo destino más visitado es River North (en color naranja) con poco menos de 10000 viajes para los días 15 y 16 del mes de Noviembre. Los siguientes destinos son Streetville (en verde) y West Loop  (en rojo) con entre 4000 y 8000 viajes, finalmente encontramos a O'hare, Lake View, Grant Park, Museum Campus, Gold Coast y Sheffield & DePaul con viajes menores a 4000.
# </div>

# In[12]:


top_10_companies = sql_result_01.head(10)
sns.barplot(data= top_10_companies, x='company_name', y='trips_amount')
plt.ylabel('Viajes promedio')
plt.xlabel('Compañías de taxi')
plt.xticks(rotation=90)
plt.title('Cantidad de viejes hechos por las compañías de taxis')


# <div class="alert alert-block alert-info" style="background-color: #DDB0DD; border-color: #800080; color: #000000;">
#     <b></b> <a class="tocSkip"></a>
#     La empresa de taxis Flas Cab (en azul) tuvo la mayor cantidad de viajes para el 15 y 16 de Noviembre con poco menos de 20000 viajes. Para las compañías siguientes se observa un salto drástico en la cantidad de viajes que tuvieron menos de 12500 pero más de 5000. Por lo que, aparentemente, la compañía más popular o con mayor cantidad de unidades es Flash Cab.
# </div>

# <div class="alert alert-block alert-info" style="background-color: #DA70D6; border-color: #800080; color: #8B008B;">
#     <b>PRUEBAS DE HIPÓTESIS</b> <a class="tocSkip"></a>
#     
# </div>

# <div class="alert alert-block alert-info" style="background-color: #DDB0DD; border-color: #800080; color: #000000;">
#     <b>♣</b> <a class="tocSkip"></a>
#     En esta sección, por practicidad, hice un análisis exploratorio de datos aparte para la sección 07 de resultados. Los valores de fecha y hora de la columna 'start_ts' fueron convertidos a todatetime para poder utilizar los días.
# </div>

# In[13]:


sql_result_07 = pd.read_csv('/datasets/project_sql_result_07.csv')


# In[14]:


print(sql_result_07.info())

print(sql_result_07.sample(20))


# In[15]:


sql_result_07.isna().sum()


# In[16]:


sql_result_07.duplicated().sum()


# In[17]:


sql_result_07['start_ts'].value_counts(dropna=False).sort_index()


# In[18]:


sql_result_07 ['start_ts'] = pd.to_datetime(sql_result_07['start_ts'])


# In[19]:


sql_result_07['weather_conditions'].value_counts(dropna=False).sort_index()


# In[20]:


sql_result_07['duration_seconds'].value_counts(dropna=False).sort_index()


# <div class="alert alert-block alert-info" style="background-color: #DDB0DD; border-color: #800080; color: #000000;">
#     <b></b> <a class="tocSkip"></a>
#     Los valores duplicados de este DataFrame para la columna 'start_ts' indican que se realizó más de un viaje a la misma hora, y los valores duplicados para la columna 'duration_seconds' indica que diferentes viajes tuvieron la misma duración (en segundos). Convertiré la duración de los viajes para visualizarlos a escala.
# </div>
# 

# <div class="alert alert-block alert-info" style="background-color: #AFEEEE; border-color: #800080; color: #000000;">
#     <b>H0</b> <a class="tocSkip"></a>
#     "La duración promedio de los viajes desde el Loop hasta el Aeropuerto Internacional O'Hare no cambia los sábados lluviosos"
# </div>
# 
# <div class="alert alert-block alert-info" style="background-color: #40E0D0; border-color: #800080; color: #000000;">
#     <b>H1</b> <a class="tocSkip"></a>
#     "La duración promedio de los viajes desde el Loop hasta el Aeropuerto Internacional O'Hare cambia los sábados lluviosos"
# </div>
# 

# In[21]:


sat_bad_weather = sql_result_07[(sql_result_07['start_ts'].dt.dayofweek == 5) & (sql_result_07['weather_conditions'] == 'Bad')] 
sat_good_weather = sql_result_07[(sql_result_07['start_ts'].dt.dayofweek == 5) & (sql_result_07['weather_conditions'] == 'Good')] 
interested_value= sat_good_weather ['duration_seconds'].mean()     
print("Duración de viajes con malas condiciones climáticas para el sábado:")
print(sat_bad_weather['duration_seconds'])

alpha = 0.05
results = st.ttest_1samp(sat_bad_weather['duration_seconds'], interested_value)

print(f"P-value:", results.pvalue)


# <div class="alert alert-block alert-info" style="background-color: #DDB0DD; border-color: #800080; color: #000000;">
#     <b></b> <a class="tocSkip"></a>
#     Para probar esta hipótesis utilicé la librería scipy con la función de stats. Opté por realizar una prueba de dos colas ya que normalemente las hipótesis se construyen sobre las medias para poder demostrar que las medias de dos poblaciones son iguales o diferentes (como en este caso),  para poder realizar una t de student utilicé un valor fijo como la media de mi población "control" y a partir de ahí realicé la comparación la cual demuestra (con un a p < 0.05) que la media de la población experimental o en este caso con lluvia es estadísticamente diferente en cuanto al tiempo, por lo que se rechaza H0.
# </div>

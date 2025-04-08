#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A52A2A;">
#     <b>Introduccion:</b> <a class="tocSkip"></a>
#     La tienda online Ice necesita un modelo para planificar campañas publicitarias, esto se puede lograr analizando reseñas de usuarios y expertos de videojuegos sobre la popularidad de plataformas y consolas. 
# </div>

# In[ ]:


from scipy import stats as st
from scipy.stats import ttest_ind
from scipy.stats import levene
from matplotlib import pyplot as plt
import math as mt
import numpy as np
import pandas as pd 
import seaborn as sns 


# In[2]:


games_info = pd.read_csv('/datasets/games.csv')


# In[3]:


print(games_info.info())


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A52A2A;">
#     <b>Preparar los datos</b> <a class="tocSkip"></a>
#     
# </div>

# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Código para reemplazar los nombres de las columnas y que queden todos en minúsculas:
# </div>

# In[4]:


new_col_names=[]
for old_name in games_info.columns:
    name_lowered=old_name.lower()
    new_col_names.append(name_lowered)
    
games_info.columns=new_col_names
print(games_info.info())


# La columna 'Year_of_release' podría trabajarse como todatetime y la columna 'user_score' debería ser float64. Este último debe ser procesado para poderlo convertir al formato adecuado. Más adelante se muestra el código.

# In[5]:


games_info['year'] = pd.to_datetime(games_info['year_of_release'], format='%Y')
print(games_info['year'].dtype)


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Código para revisar los valores ausentes de todo el dataframe:
# </div>

# In[6]:


games_info.isna().sum()


# In[7]:


games_info.duplicated().sum()


# In[8]:


print(games_info.head(20))


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Código para revisar el tipo de valores ausentes de las columnas con valores ausentes mostradas anteriormente:
# 
# </div>
# 

# In[9]:


print(games_info['name'].unique())
print(games_info['name'].value_counts(dropna=False).sort_index())


# In[10]:


print(games_info['year_of_release'].unique())
print(games_info['year_of_release'].value_counts(dropna=False).sort_index())


# In[11]:


print(games_info['genre'].unique())
print(games_info['genre'].value_counts(dropna=False).sort_index())


# In[12]:


print(games_info['critic_score'].unique())
print(games_info['critic_score'].value_counts(dropna=False).sort_index())


# In[13]:


print(games_info['user_score'].unique())
print(games_info['user_score'].value_counts(dropna=False).sort_index())


# In[14]:


print(games_info['rating'].unique())
print(games_info['rating'].value_counts(dropna=False).sort_index())


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Diversas columnas tienen valores ausentes los cuales se abordan de distinta manera: 
#     Los valores ausentes de las columnas 'name', 'genre' y 'year_of_release' representan menos del 2% del total de los datos por lo que no será un porblema dejarlos vacíos. 
#     Por otro lado, los valores ausentes de las columnas 'critic_score', 'user_score' y 'rating' representan cerca del 50% de los datos por lo tanto no deben ignorarse. Ambos scores podrían rellenarse con la mediana en caso de que tengan valores exrtemos, si no, con la media. En el caso del rating, se podrian rellenar de acuerdo al género.
# </div>
#     
# 

# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Se calcula la mediana de los valores presentes de cada columna para poder rellenar los ausentes y se crean columnas nuevas:
# </div>

# In[15]:


critics_dropna = games_info['critic_score'].dropna().median()
print(critics_dropna)
print(games_info['critic_score'].dropna().describe())
games_info['critic_score_filled'] = games_info['critic_score'].fillna(71.0)
print(games_info.head(20))


# In[16]:


games_info['critic_score'].isna().groupby(games_info['year']).mean().plot(kind='line', color='m')
plt.xlabel('Año de lanzamiento')
plt.ylabel('Frecuencia')
plt.title('Porcentaje de videojuegos con critic score de 1980 a 2015')
plt.tight_layout()
plt.legend()
plt.grid(axis='y')


# In[17]:


users_not_tbd = games_info['user_score'].replace('tbd', pd.NA)
print(users_not_tbd.value_counts(dropna=False))
users_notna = users_not_tbd.dropna()
print(users_notna)
users_median = users_notna.astype('float').median()
print(users_median)
print(users_notna.astype('float').describe())


# In[18]:


games_info['user_score'].isna().groupby(games_info['year']).mean().plot(kind='line', color='c')
plt.xlabel('Año de lanzamiento')
plt.ylabel('Frecuencia')
plt.title('Porcentaje de videojuegos con user score de 1980 a 2015')
plt.tight_layout()
plt.legend()
plt.grid(axis='y')


# In[19]:


users_not_tbd = users_not_tbd.replace(pd.NA, '7.5')
print(users_not_tbd.value_counts(dropna=False))
games_info ['user_score_filled'] = users_not_tbd
print(games_info.head(20))


# In[20]:


games_info['rating'].isna().groupby(games_info['year']).mean().plot(kind='line', color='r')
plt.xlabel('Año de lanzamiento')
plt.ylabel('Frecuencia')
plt.title('Porcentaje de videojuegos con rating de 1980 a 2015')
plt.tight_layout()
plt.legend()
plt.grid(axis='y')


# In[21]:


games_info['rating_filled'] = games_info['rating']
def rating_list (genre_value,rating_value):
    games_info.loc[(games_info.genre == genre_value) & (games_info.rating_filled.isnull()), 'rating_filled'] = rating_value
rating_list("Action","E")
rating_list("Adventure","E")
rating_list("Fighting","T")
rating_list("Misc", "E")
rating_list("Platform", "E")
rating_list("Puzzle", "E")
rating_list("Racing", "E")
rating_list("Role-Playing", "E")
rating_list("Shooter", "T")
rating_list("Simulation", "E")
rating_list("Sports", "E")
rating_list("Strategy", "E")

print(games_info.sample(20))


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Los valores ausentes de las columnas 'critic_score', 'user_score' y 'rating' se trataron de diferente manera; la columna 'critic_score' tenía sólo 'NaNs' los cuales fueron sustituidos por la mediana ya que la media y la mediana fueron diferentes entre sí. La columna 'user_score' tenía 'NaNs' y 'tbds', los 'tbd' fueron reemplazados por 'NaN' y luego rellenedos con la mediana total, en este caso la media y la mediana eran iguales, así que cualquiera de los dos datos era valido utilizar. Para la última columna ('rating') rellené los valores ausentes de acuerdo al genero.
#     Con las gáficas lineales del comportamiento de los 'scores' por año vemos que antes del 2000 los videojuegos no tenían ninguna de las tres calificaciones, posterior al 2000 los juegos comiezan a tener calificaciones, con esto podria deducirse que los valores ausentes despues de este año se deben a la baja popularidad de los videojuegos, el porqué de esta baja popularidad es mas dificil de explicar pues se puede deber a la baja calidad del juego, a la deficiente mercadotecnia que se empleó para dar a conocer el juego y convencer de adquirirlo, etc. 
# </div>

# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Se calculan las ventas totales para cada juego:
# </div>

# In[22]:


games_info['total_sales'] = games_info['na_sales'] + games_info['eu_sales'] + games_info['jp_sales'] + games_info['other_sales']
print(games_info.head(20))


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A52A2A;">
#     <b>Analizar datos</b> <a class="tocSkip"></a>
#     
# </div>

# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Videojuegos lanzados por año:
# </div>

# In[23]:


games_per_year = games_info.groupby('year')['name'].count().reset_index()
print(games_per_year)
games_info.pivot_table(index='year', columns='genre', values='critic_score', aggfunc='mean').plot(kind='line')
plt.xlabel('Año')
plt.ylabel('Número de videojuegos')
plt.title('Promedio de videojuegos lanzados por año por género')
plt.legend()
plt.grid(axis='y')


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Los años que tuvieron mayores ventas fue entre 2006 y 2011.
# </div>

# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Ventas por plataforma:
# </div>

# In[24]:


platform_sales = games_info.groupby('platform')[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum()
platform_sales['platform_total_sales'] = platform_sales.sum(axis=1)
print(platform_sales.sort_index(ascending=False))


# In[25]:


start_year = 1980
end_year = 1993
mask = (games_info['year'].dt.year < start_year) | (games_info['year'].dt.year > end_year)

games = games_info[mask]
popular_platforms = games.groupby('platform')['total_sales'].sum().sort_values().tail(10)
print(popular_platforms)


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Las plataformas que tuvieron mayores ventas fueron: PS2, PS3, X360, Wii y DS.
# </div>

# In[26]:


popular_platforms = list(games.groupby('platform')['total_sales'].sum().sort_values().tail(5).index)

(
    games_info[games_info['platform'].isin(popular_platforms)]
    .pivot_table(index='year',
                columns='platform',
                values='total_sales',
                aggfunc='sum')
    .plot(kind='line')
)
plt.xlabel('Año')
plt.ylabel('Millones de unidades')
plt.title('Ventas totales de las plataformas más populares')
plt.legend()
plt.grid(axis='y')


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Distribución basada en los datos de cada año para las 5 plataformas más populares:
# </div>

# In[27]:


top_platforms = games[games['platform'].isin(popular_platforms)]

plt.figure(figsize=(12, 8))
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title('Distribución de ventas cada año (Plataformas populares)')

plt.ylim(0, 2.0)  

sns.boxplot(data=top_platforms, x='year_of_release', y='total_sales', hue='platform', palette='Set3', linewidth=1.5)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.xlabel('Año')
plt.ylabel('Ventas Totales')
plt.show()


# In[28]:


sales_by_year_platform = games.groupby(['year', 'platform'])['total_sales'].sum().reset_index()

sales_by_year_platform[sales_by_year_platform['platform'].isin(popular_platforms)][['total_sales','platform']].boxplot(
    column = 'total_sales',
    by = 'platform',
)
plt.title('Distribución de ventas por año (Plataformas populares)')
plt.xlabel('Plataforma')
plt.ylabel('Ventas totales')
plt.show()


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Las plataformas que eran populares pero ya no tienen ventas: PS, PS2, Wii y DS.
# </div>

# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Las plataformas tardan al rededor de 6 años en aparecer y desaparecer:
# </div>

# In[29]:


nintendo_platforms = ['NES','SNES','N64','GC','Wii','WiiU']
games_info[games_info['platform'].isin(nintendo_platforms)].pivot_table(index='year_of_release', 
                                                                        columns='platform', 
                                                                        values='total_sales', 
                                                                        aggfunc='sum').plot(kind='line')
plt.title('Ventas de las platafomas de Nintendo a lo largo del tiempo')
plt.xlabel('Año')
plt.ylabel('Ventas Totales')


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     La información más relevante la encontramos a partir del año 1994, de tal modo que ignoraré de 1993 hacia atrás:
# </div>

# In[30]:


start_year = 1980
end_year = 1993
mask = (games_info['year'].dt.year < start_year) | (games_info['year'].dt.year > end_year)

games = games_info[mask]

print(games)
print(games['year'].value_counts(dropna=False))


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Plataformas líderes en ventas por año: PS2, PS3, X360, Wii y DS.
# </div>

# In[31]:


games_info.pivot_table(index='year', columns='platform', values='total_sales', aggfunc='sum').plot(kind='line')

plt.xlabel('Año')
plt.ylabel('Millones de unidades')
plt.title('Total de ventas por plataforma por año')
plt.legend()

plt.grid(axis='y')


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Plataformas que han vendido durante más años consecutivos (rentables):
# </div>

# In[32]:


games_info.groupby('platform')['year'].nunique().sort_values(ascending=False)


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Dsitribución de ventas por plataforma:
# </div>

# In[33]:


games[games['platform'].isin(popular_platforms)][['total_sales','platform']].boxplot(
    column = 'total_sales',
    by = 'platform',
)
plt.xlabel('Plataforma')
plt.ylabel('Ventas totales')
plt.title('Distribución de ventas por plataforma')
plt.ylim(0,2.5)
plt.show()


# In[34]:


print(games.info())


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     La distribución de las ventas es mayor para PS3 y X360, la plataforma cuya media es menor es DS 
# </div>

# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Correlación entre reseñas y ventas:
# </div>

# In[35]:


games.query("platform == 'PS3'")[['total_sales', 'critic_score_filled']].corr()


# In[36]:


x = games['critic_score_filled']
y = games['total_sales']

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', alpha=0.5)

plt.title('Ventas vs Critic_Score')
plt.xlabel('Critic_Score')
plt.ylabel('Ventas Totales')


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Existe una correlacion significativa entre las ventas para la plataforma PS3 y su 'critic_score'. De acuerdo a la gráfico de dispersión existe una relacion positiva entre 'critic_score' y las ventas, de tal forma que entre más alto es el score más ventas tienen los videojuegos. 
# </div>

# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Venta del mismo juego en otra plataforma:
# </div>

# In[37]:


games.groupby('name')['platform'].nunique().sort_values(ascending=False)


# In[38]:


juegos = ['Need for Speed: Most Wanted', 'LEGO Marvel Super Heroes', 'FIFA 14', 'Ratatouille', 'The LEGO Movie Videogame']
games_filtrados = games[games['name'].isin(juegos)]

print(games_filtrados[['name', 'platform', 'total_sales']])

plataformas = games_filtrados['platform'].unique()
num_juegos = len(juegos) 

fig, axs = plt.subplots(num_juegos, figsize=(10, 10), sharex=True, sharey=True)

for i, juego in enumerate(juegos):
    data_juego = games_filtrados[games_filtrados['name'] == juego]
    x = data_juego['platform']
    y = data_juego['total_sales']
    axs[i].bar(x, y, color=['blue', 'green', 'red', 'purple', 'orange'])
    axs[i].set_title(juego)
    
fig.suptitle('Comparación de Ventas por Plataforma para Juegos Específicos')
plt.xlabel('Plataforma')
plt.ylabel('Ventas Totales (millones)')
plt.tight_layout()

plt.show()


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Los primeros 5 juegos que estan en más de 7 plataformas en general tuvieron más ventas para Xbox360 depués PS3, 3DS y finalmente PS4.
# </div>

# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Distribución por género:
# </div>

# In[39]:


games.boxplot(column='total_sales',
             by='genre',
             )
plt.ylim(0,2.5)
plt.xticks(rotation=45)
plt.ylabel('Género')
plt.xlabel('Ventas Totales (millones)')
plt.title('Distribucion de las Ventas por Género')
plt.show()


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Aparentemente los videojuegos más rentables son en primer lugar: Platform, en segundo lugar: Shooter y en tercer lugar: fighting o sports. Los videojuegos menos rentables son Adventure, Puzzle y Strategy.
# </div>

# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A52A2A;">
#     <b>Perfil de usuario por region:</b> <a class="tocSkip"></a>
#     
# </div>

# In[40]:


regionNA = "na_sales"

platforms_sales = games.groupby("platform")[regionNA].sum()
top_5_platforms_na = platforms_sales.sort_values().tail(5).reset_index()
top_5_platforms_na['market_share'] = top_5_platforms_na[regionNA] / platforms_sales.sum()
top_5_platforms_na


# In[41]:


top_5_platforms_na['market_share'].sum()


# In[42]:


regionEU = 'eu_sales'

platforms_sales = games.groupby("platform")[regionEU].sum()
top_5_platforms_eu = platforms_sales.sort_values().tail(5).reset_index()
top_5_platforms_eu['market_share'] = top_5_platforms_eu[regionEU] / platforms_sales.sum()
top_5_platforms_eu


# In[43]:


top_5_platforms_eu['market_share'].sum()


# In[44]:


regionJP = 'jp_sales'

platforms_sales = games.groupby("platform")[regionJP].sum()
top_5_platforms_jp = platforms_sales.sort_values().tail(5).reset_index()
top_5_platforms_jp['market_share'] = top_5_platforms_jp[regionJP] / platforms_sales.sum()
top_5_platforms_jp


# In[45]:


top_5_platforms_jp['market_share'].sum()


# In[46]:


games_info['ratings_summed'] = np.where(
    games_info['rating_filled'].isin(['EC','K-A','RP','AO']),
    "Other",
    games_info['rating_filled']
)
print(games_info['ratings_summed'])


# In[47]:


for region in ['na_sales', 'eu_sales', 'jp_sales']:
    
    fig, axes = plt.subplots(ncols=3, figsize=(15, 4)) 
    
    sales_by_platform = games_info.groupby('platform')[region].sum()
    top_platforms = sales_by_platform.sort_values().tail(5).reset_index()
    top_platforms['market_share'] = top_platforms[region] / sales_by_platform.sum()
    
    top_genres = games_info.groupby('genre')[region].sum().sort_values().tail(5).reset_index()
    
    top_ratings = games_info.groupby('ratings_summed')[region].count().sort_values().reset_index()
    
    fig.suptitle(f"Ventas en la región {region}", y=0.96)
    
    axes[0].set_title('Top 5 Plataformas con mayor Market Share')
    top_platforms.plot(kind='bar', x='platform', y= 'market_share', ax=axes[0])
    
    axes[1].set_title('Top 5 Géneros con mayores ventas')
    top_genres.plot(kind='bar', x='genre', y=region, ax=axes[1], color = 'm')
    
    axes[2].set_title('Top Ratings con mayores ventas')
    top_ratings.plot(kind='bar', x='ratings_summed', y=region, ax=axes[2], color='g')
    
    fig.tight_layout(pad=1)


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Lo sobresaliente de estos gráficos es que podemos ver resumido las similitudes y diferencias del comportamiento de las ventas por región. Así, podemos ver que en Norte América fue más popular el Xbox360 por lo tanto el Market Share en mayor gracias a esta consola, mientras que en Europa fue la PS2 y en Japón el DS. 
#     El género que tuvo mayores ventas en NA fue 'Action' igual que en EU, sin embargo, en JP esto fue diferente siendo el género de Role-Playing el que recaudó mayores ventas. Finalmente, en cuanto al Rating que se le atribuye una mayor popularidad podemos decir que para las tres regiones es E. 
# </div>

# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A52A2A;">
#     <b>Pruebas de hipótesis:</b> <a class="tocSkip"></a>
#     
# </div>

# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>H1:</b> <a class="tocSkip"></a> Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son diferentes.
# </div>

# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>H0:</b> <a class="tocSkip"></a> Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son las mismas.
# </div>

# In[48]:


xone_scores = games[games['platform'] == 'XOne']['user_score_filled'].astype(float)
pc_scores = games[games['platform'] == 'PC']['user_score_filled'].astype(float)

statistic, p_value = levene(xone_scores, pc_scores)

print(p_value)

alpha = 0.05
t_statistic, p_value = ttest_ind(xone_scores, pc_scores, equal_var=True)

print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Utilicé primero una prueba de Levene para determinar estadísticamente la diferncia entre las varianzas la cual arrojó que no habia diferencias significativas. Con lo anterior determiné utilizar un ttest para muestras independientes ya que asume que sus varianzaz son idénticas. Así, para la prueba de hipótesis establecí un equal_var=True con lo anterior y a pesar de haber rellenado los tbd y NaNs con la mediana obtuve que sí existen diferencias significativas entre la media de la calificación de los usuarios para la plataforma XOne contra PC con un valor de p < 0.05. 
# 
# </div>

# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>H1:</b> <a class="tocSkip"></a>Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes.
# 
# </div>
# 
# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>H0:</b> <a class="tocSkip"></a>Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son iguales.
# </div>

# In[49]:


action_scores = games[games['genre'] == 'Action']['user_score_filled'].astype(float)
sports_scores = games[games['genre'] == 'Sports']['user_score_filled'].astype(float)

statistic, p_value = levene(action_scores, sports_scores)

print(p_value)

alpha = 0.05
t_statistic, p_value = ttest_ind(action_scores, sports_scores, equal_var=True)

print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")


# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b>♣</b> <a class="tocSkip"></a>
#     Para esta prueba de hipotesis también utilicé un ttest_ind para muestras independientes bajo el mismo criterio de igualdad de las varianzas, esto lo corroboré con una prueba de Levene y podemos conlcuir que no hay diferencias significativas entre la calificación promedio de los videojuegos del género 'Action' comparado con los videojuegos del género 'Sports'. Por lo tanto, no se rechaza la hipótesis nula con un valor de p > 0.05. En este caso, este resultado pudo verse influido por la modificacion de los NaNs y tbd con la mediana lo cual igualó las calificaciones para ambos géneros. 
# </div>

# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A52A2A;">
#     <b>Conclusión general:</b> <a class="tocSkip"></a> 
# </div>

# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b> </b> <a class="tocSkip"></a> 
#     Para poder analizar los datos es importante tener homogeneidad en el formato de los nombres de las columnas, al igual que en el formato de los datos. Es probable que nos encontremos con valores ausentes o diferentes los cuales deben de valorarse para encontrar la mejor manera de tratarlos, ya sea rellenándolos o ignorándolos dependiendo de la información que proveen. También es immportante mantener congruencia en los gráficos siendo lo más explícitos posibles en cada uno de ellos para entender lo que se quiere transmitir.
# </div>
# 
# <div class="alert alert-block alert-info" style="background-color: #E6CBDD; border-color: #3E3C3C; color: #A67F78;">
#     <b> </b> <a class="tocSkip"></a> 
#     En cuanto a los resultados obtenidos del análisis del dataframe de videojuegos podemos decir que de todos las plataformas evaluadas existen unas que tuvieron más exito que otras, al igual que de los videojuegos. Hay parámetros que son determinantes en el éxito de un videojuego como la plataforma, el género y a quien va dirigido. Finalmente, con estos elementos podemos hacer una proyección del comportamiento de las ventas de videojuegos nuevos para años posteriores.
# </div>

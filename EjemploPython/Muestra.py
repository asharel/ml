
# coding: utf-8

# In[57]:

#Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# In[46]:

# Read Dataset
dataset = pd.read_csv('C:\\Users\Ana\Documents\home\Master-TIC\Curso_2016-17\wine.csv',delimiter=',',header=0)
dataset


# In[47]:

# Header
header = []
for row in dataset:
    header.append(row)
header


# In[48]:

print(dataset.describe()) #Descripción de los datos


# In[49]:

correlation=dataset.corr() #Correlation Matrix
correlation


# In[50]:

# Display the correlation matrix with a specified figure number and a bluescale
# colormap
plt.figure()
plt.matshow(correlation, fignum=1, cmap=plt.cm.Blues)
plt.ylabel("Attribute Index")
plt.show()


# In[51]:

#### Scatter Matrix Plot

plt.figure()
from pandas.tools.plotting import scatter_matrix
scatter_matrix(dataset, alpha=0.3, figsize=(20, 20), diagonal='kde')
plt.show()


# In[52]:

#### Histogram Matrix Plot

plt.figure()
dataset.hist(xlabelsize=0.5, ylabelsize=0.2,figsize=(10,10))
plt.xlabel("Data")
plt.show()


# In[ ]:




# In[53]:

### Histogram of Alcohol variable

plt.figure(figsize=(8,4))
plt.hist(dataset[dataset.Wine==1].Alcohol, 30, facecolor='r')
plt.hist(dataset[dataset.Wine==2].Alcohol, 30, facecolor='g')
plt.hist(dataset[dataset.Wine==3].Alcohol, 30, facecolor='b')
plt.title('Histogram of Alcohol')
plt.legend(['Wine A','Wine B','Wine C'])
plt.xlabel("Alcohol")
plt.grid(True)
plt.show()


# In[54]:


### Scatter Plot Alcohol vs Ash

plt.figure()
plt.scatter(dataset[dataset.Wine==1].Alcohol,dataset[dataset.Wine==1].Ash, color='red')
plt.scatter(dataset[dataset.Wine==2].Alcohol,dataset[dataset.Wine==2].Ash, color='blue')
plt.scatter(dataset[dataset.Wine==3].Alcohol,dataset[dataset.Wine==3].Ash, color='green')
plt.title('Scatter Plot: Alcohol vs Ash')
plt.xlabel('Alcohol')
plt.ylabel('Ash')
plt.legend(['A','B','C'])
plt.show()


# División de la muestra en Train y Test de forma aleatoria

# In[55]:

def dividir_ent_test(dataframe, porcentaje=0.7):
    """ 
    Función que divide un dataframe aleatoriamente en entrenamiento y en test.
    Recibe los siguientes argumentos:
    - dataframe: DataFrame que vamos a utilizar para extraer los datos
    - porcentaje: porcentaje de patrones en entrenamiento
    Devuelve:
    - train: DataFrame con los datos de entrenamiento
    - test: DataFrame con los datos de test
    """
    mascara = np.random.rand(len(dataframe)) < porcentaje
    train = dataframe[mascara]
    test = dataframe[~mascara]
    return train, test

wine_train, wine_test = dividir_ent_test(dataset)
print ("wine_train", wine_train.shape,"\n")
print ("wine_test",wine_test.shape,"\n")


# In[56]:

### Scatter Plot Alcohol vs Ash

plt.figure(figsize=(8,6))
plt.subplot(121)
plt.scatter(wine_train[wine_train.Wine==1].Alcohol,wine_train[wine_train.Wine==1].Ash, color='red')
plt.scatter(wine_train[wine_train.Wine==2].Alcohol,wine_train[wine_train.Wine==2].Ash, color='blue')
plt.scatter(wine_train[wine_train.Wine==3].Alcohol,wine_train[wine_train.Wine==3].Ash, color='green')
plt.title('Alcohol vs Ash; Train dataset\n',fontsize=10)
plt.xlabel('Alcohol',fontsize=10)
plt.ylabel('Ash')
plt.legend(['A','B','C'])
plt.xlim(10,16)
plt.ylim (1,3.5)

plt.subplot(122)
plt.scatter(wine_test[wine_test.Wine==1].Alcohol,wine_test[wine_test.Wine==1].Ash, color='red')
plt.scatter(wine_test[wine_test.Wine==2].Alcohol,wine_test[wine_test.Wine==2].Ash, color='blue')
plt.scatter(wine_test[wine_test.Wine==3].Alcohol,wine_test[wine_test.Wine==3].Ash, color='green')
plt.title('Alcohol vs Ash; Test dataset \n',fontsize=10)
plt.xlabel('Alcohol', fontsize=10)
plt.ylabel('Ash')
plt.legend(['A','B','C'])
plt.xlim(10,16)
plt.ylim (1,3.5)
plt.show()


# In[ ]:




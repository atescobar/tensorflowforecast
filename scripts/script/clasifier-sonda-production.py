
# coding: utf-8

# ## Tratamiendo de los datos de SONDA

# In[1]:



import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import math

from helper_fn import preprocess_features, preprocess_new_features


# ## Preprocesamiento de los datos
# 
# Primero, cargamos los datos desde un archivo CSV

# In[2]:


oportunidades = pd.read_csv("oportunidades2.csv")
to_predict_file= pd.read_csv("Libro14.csv")
to_predict_file.drop(['Unnamed: 10'], axis=1, inplace=True)


# In[3]:



#to_predict_file


# La siguiente función toma el archivo ``oportunidades.csv`` y selecciona los atributos utiles para este analisis

# Buscamos los dos sets de filial crm y filial no crm

# In[4]:


filial_crm, filial_nocrm = preprocess_features(oportunidades)
to_predict_features = preprocess_new_features(to_predict_file)

n = len(filial_crm.columns)

filial_crm.drop(['cmtx_unidaddenegocioname', 'Duracion_dias', 'Estado_Perdido'], axis=1, inplace=True)
to_predict_features.drop(['cmtx_unidaddenegocioname'], axis=1, inplace=True)


# In[5]:


to_predict_features.dtypes


# ## Sets de prueba y de entrenamiento
# Dividimos la muestra en prueba y entrenamiento, dejando de entrenamiento los datos del 2018. Primero trabajaremos solo con las filiales que son parte del crm, para las cuales tenemos datos de clasificación sensibles.

# In[6]:



thisyear = pd.to_datetime('today').year
year_str = '1/1/'+str(thisyear)
thisyear = pd.to_datetime(year_str)
#ts = thisyear - pd.DateOffset(years=1)
ts = thisyear - pd.DateOffset(years=1)



#to_predict_features = filial_crm.loc[:, filial_crm.columns != 'Estado_Ganado'].loc[filial_crm.index > thisyear]
to_predict_id = to_predict_features['Opportunityid']
to_predict_features.drop(['Opportunityid'],axis=1, inplace=True)
to_train_features = filial_crm.loc[:, filial_crm.columns != 'Estado_Ganado'].loc[filial_crm.index <= thisyear]
to_train_target = filial_crm['Estado_Ganado'].loc[filial_crm.index <= thisyear]
to_train_features.drop(['opportunityid'],axis=1, inplace=True)



train = to_train_features
target = to_train_target


#montototal = filial_crm[['totalamount_base', 'Estado_Ganado']]
#monto = filial_crm[['totalamount_base', 'Estado_Ganado']]
#montototal = montototal.loc[montototal['Estado_Ganado'] == 1]
#montototal = montototal.loc[montototal.index >= thisyear]
#montototal = montototal.loc[montototal.index < thisyear]
#filial_crm.drop(['totalamount_base'],axis=1, inplace=True)
#montototal.drop(['Estado_Ganado'],axis=1, inplace=True)


# ## Crear el modelo
# 

# In[7]:




def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(10).repeat().batch(batch_size)
    return dataset




def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    return dataset



# In[8]:


feature_columns = []

for key in train.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))


# In[9]:


classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[6 ,],
    n_classes=2,
    )


# # Train the model

# In[21]:


batch_size = 100
train_steps = 800
 
classifier.train(input_fn = lambda: train_input_fn(train, target, batch_size), steps = train_steps)
#eval_result = classifier.evaluate(input_fn = lambda: eval_input_fn(to_predict_features, None,batch_size))
#print(eval_result)

#accuracy_score = classifier.evaluate(input_fn=lambda:eval_input_fn(features_test,labels=target_test,batch_size=batch_size))['accuracy']
#print('\nTest Accuracy: {0:f}%\n'.format(accuracy_score*100))


# # Predicciones
# 
# Ahora vemos como se comporta en la predicción de cierre de oportunidad

# In[22]:



predictions = classifier.predict(input_fn=lambda:eval_input_fn(to_predict_features,labels=None, batch_size=batch_size))

template = ('\nPrediction is {} with ({:.1f}%) probability')

val = []
val2 = []
val3 = []
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0] 
    if(class_id==0):
        probability = 1-pred_dict['probabilities'][class_id]
    if(class_id==1):
        probability = pred_dict['probabilities'][class_id]
    val2.append(class_id)
    val3.append(probability)
    
    #print(template.format(class_id, 100 * probability))




# In[23]:


val1 = to_predict_id.as_matrix()

df = pd.DataFrame({'opt_id': val1, 'class_id': val2, 'prob':val3})
df['opt_id'] = df['opt_id'].astype(int)
df = df.set_index('opt_id')

df.to_csv('results.csv')

print("\nPredictions stored in results.csv")
print('Total Closed oportunities')
print(df['class_id'].sum())

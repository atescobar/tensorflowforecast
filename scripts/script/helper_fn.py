#Preprocesamiento de las caracteristicas 

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import math



#def get_data(filename):
#    df = pd.read_csv(filename)
#    if (df['cmtx_fechacierreoportunidad'].isnull()
#    df1, df2 = preprocess_features(df)
#    return df1, df2

def ventas_totales_por_dia(df):

    ganado = df.loc[df['Estado_Ganado'] == 1]
    ganado = ganado.groupby(ganado.index.to_period("M")).sum()
    ganado = ganado/1e10
    
    
    return ganado

def preprocess_features(raw_data):
    #Seleccionar atributos
    selection = ['opportunityid','createdon', 'cmtx_filialname', 'cmtx_fechacierreoportunidad', 'totalamount_base', 'cmtx_naturalezadeoportunidad_displayname','statecode_displayname', 'cmtx_unidaddenegocioname', 'cmtx_etapadeproceso_displayname', 'cmtx_commitedtoforecast']
    selected_features = raw_data[selection]
    #Eliminar valores nulos
    selected_features = selected_features.dropna()
   
    #convertir a fechas para manipualar

    selected_features['cmtx_fechacierreoportunidad'] = pd.to_datetime(selected_features['cmtx_fechacierreoportunidad'])
    selected_features['createdon'] = pd.to_datetime(selected_features['createdon'])
    selected_features['Duracion'] = selected_features['cmtx_fechacierreoportunidad'] - selected_features['createdon']
    
    #Contar los días que una oportunidad está abierta
    aux = []
    for days in selected_features['Duracion']:
        val = str(days).split(' ')
        aux.append(val[0].replace("-", ""))
        
    for i in range(0, len(aux)-1):
        aux[i] = int(aux[i])

        if (math.isnan(aux[i])):
            print(aux[i])
            aux[i] = 0
            
    selected_features = selected_features.loc[selected_features['cmtx_filialname'] == 'Sonda Chile']
        
    selected_features['Duracion_dias'] = pd.Series(aux, dtype='int32')
    
    #Convertir monto total a numero flotante.
    selected_features['totalamount_base'] = selected_features['totalamount_base'].str.replace(",","")
    selected_features['totalamount_base'] = selected_features['totalamount_base'].astype('float32')
    
    selected_features.drop(['Duracion'],axis=1, inplace=True)
    selected_features.drop(['createdon'],axis=1, inplace=True)
    
    
    ## One-Hot Encoding
#    selected_features.cmtx_naturalezadeoportunidad_displayname = selected_features.cmtx_naturalezadeoportunidad_displayname.str.replace("ó",'o')
#    selected_features.cmtx_naturalezadeoportunidad_displayname = selected_features.cmtx_naturalezadeoportunidad_displayname.str.replace(" ",'')
#    naturaleza = pd.DataFrame({'cmtx_naturalezadeoportunidad_displayname': selected_features.cmtx_naturalezadeoportunidad_displayname})    
#    selected_features = pd.concat([selected_features, pd.get_dummies(naturaleza, prefix='naturaleza')],axis=1)
    selected_features.drop(['cmtx_naturalezadeoportunidad_displayname'],axis=1, inplace=True)
    
   
    #selected_features.cmtx_etapadeproceso_displayname = selected_features.cmtx_etapadeproceso_displayname.str.replace("ó",'o')
#    etapa = pd.DataFrame({'etapa': selected_features.cmtx_etapadeproceso_displayname})    
#    selected_features = pd.concat([selected_features, pd.get_dummies(etapa, prefix='etapa')],axis=1)
    selected_features.drop(['cmtx_etapadeproceso_displayname'],axis=1, inplace=True)
    
    ##### One-Hot encode para negocios relevantes o no relevantes.
    aux = []
    for value in selected_features.totalamount_base:
        if value > 100000:
            aux.append(1)
        else:
            aux.append(0)
            
    relevante = pd.DataFrame({'relevante': aux})
    selected_features = pd.concat([selected_features, relevante],axis=1)
    
    
    #### One-Hot encode para numero de dias

    aux_quarter = []
    aux_semester = []
    aux_3quarter = []
    aux_year = []
    aux_overyear = []
    for value in selected_features['Duracion_dias']:
        if(value <= 90):
            aux_quarter.append(1) 
            aux_semester.append(0)
            aux_3quarter.append(0)
            aux_year.append(0) 
            aux_overyear.append(0) 
        elif(value > 90 and value <= 180):
            aux_quarter.append(0) 
            aux_semester.append(1)
            aux_3quarter.append(0)
            aux_year.append(0) 
            aux_overyear.append(0) 
        
        elif(value < 180 and value <= 270):
            aux_quarter.append(0) 
            aux_semester.append(0)
            aux_3quarter.append(1)
            aux_year.append(0) 
            aux_overyear.append(0) 
            
        elif(value > 270 and value <= 360):
            aux_quarter.append(0) 
            aux_semester.append(0)
            aux_3quarter.append(0)
            aux_year.append(1) 
            aux_overyear.append(0) 
            
        elif(value > 360):
            aux_quarter.append(0) 
            aux_semester.append(0)
            aux_3quarter.append(0)
            aux_year.append(0) 
            aux_overyear.append(1) 
            
    aux = [aux_quarter, aux_semester, aux_3quarter, aux_year, aux_overyear]
    
    periodOfTime = pd.DataFrame({'period_1quarter': aux[0], 'period_2quarter': aux[1], 'period_3quarter': aux[2], 'period_4quarter': aux[3], 'period_overayear': aux[4]})
    selected_features = pd.concat([selected_features, periodOfTime],axis=1)
    
    
    
    #### Ganada o perdida? #####
    estado = pd.DataFrame({'statecode_displayname': selected_features.statecode_displayname})    
    selected_features = pd.concat([selected_features, pd.get_dummies(estado, prefix='Estado')],axis=1)
    selected_features.drop(['statecode_displayname'],axis=1, inplace=True)

    
    
    #Ordenar por fecha de cierre
    selected_features = selected_features.set_index(['cmtx_fechacierreoportunidad'])
    selected_features.sort_index(inplace=True)
    
    # Separamos en dos archivos las cargas de filiales en crm y fuera de crm.
    cargas = ['MX - NOVIS','QUINTEC CHILE DISTRIB', 'CL - WIRELESS', 'CL - TECNOGLOBAL', 'CL - SOLEX', 'CL - SERVIBANCA', 'CL - NOVIS', 'CL - MICROGEO', ' CL - EDUCACIÓN', 'BR - PARS', 'BR - CTIS', 'BR - ATIVAS']


    ## Cargas de filiales.
    fr = []
    for i in range(0,len(cargas)):
        fr.append(selected_features.loc[selected_features['cmtx_unidaddenegocioname'] == cargas[i]])
    #definir dataframe con filiales fuera de CRM
    filial_nocrm = pd.concat(fr)

    for i in range(0,len(cargas)):
        selected_features = selected_features[selected_features['cmtx_unidaddenegocioname'] != cargas[i]]
    selected_features.drop(['cmtx_filialname'],axis=1, inplace=True)
    #el resto queda como filial dentro de CRM
    selected_features = selected_features.dropna()
    processed_features = selected_features 
    filial_nocrm = ventas_totales_por_dia(filial_nocrm)
    return processed_features, filial_nocrm
    
    
    
 
    
#Preprocesamiento de las caracteristicas 

    
def preprocess_new_features(raw_data):
    #Seleccionar atributos
    selection = ['Opportunityid', 'createdon', 'cmtx_filialname', 'cmtx_fechacierreoportunidad', 'totalamount_base', 'cmtx_naturalezadeoportunidad_displayname', 'cmtx_unidaddenegocioname', 'cmtx_etapadeproceso_displayname', 'cmtx_commitedtoforecast']
    selected_features = raw_data[selection]

    #Eliminar valores nulos

   
    #convertir a fechas para manipualar
    selected_features['cmtx_fechacierreoportunidad'] = pd.to_datetime(selected_features['cmtx_fechacierreoportunidad'])
    selected_features['createdon'] = pd.to_datetime(selected_features['createdon'])
    selected_features['Duracion'] = selected_features['cmtx_fechacierreoportunidad'] - selected_features['createdon']
    
    #Contar los días que una oportunidad está abierta
    aux = []
    for days in selected_features['Duracion']:
        val = str(days).split(' ')
        aux.append(val[0].replace("-", ""))
        
    for i in range(0, len(aux)-1):
        aux[i] = int(aux[i])

        if (math.isnan(aux[i])):
            print(aux[i])
            aux[i] = 0
            
   
    #selected_features['Duracion_dias'] = pd.Series(aux)
    selected_features.dropna() 
    #selected_features['Duracion_dias'] = selected_features['Duracion_dias'].astype(int)
    #Convertir monto total a numero flotante.
    selected_features['totalamount_base'] = selected_features['totalamount_base'].astype('float32')
    
    selected_features.drop(['Duracion'],axis=1, inplace=True)
    selected_features.drop(['createdon'],axis=1, inplace=True)
    
    
    ## One-Hot Encoding
#    selected_features.cmtx_naturalezadeoportunidad_displayname = selected_features.cmtx_naturalezadeoportunidad_displayname.str.replace("ó",'o')
#    selected_features.cmtx_naturalezadeoportunidad_displayname = selected_features.cmtx_naturalezadeoportunidad_displayname.str.replace(" ",'')
#    naturaleza = pd.DataFrame({'cmtx_naturalezadeoportunidad_displayname': selected_features.cmtx_naturalezadeoportunidad_displayname})    
#    selected_features = pd.concat([selected_features, pd.get_dummies(naturaleza, prefix='naturaleza')],axis=1)
    selected_features.drop(['cmtx_naturalezadeoportunidad_displayname'],axis=1, inplace=True)
    
#    selected_features.cmtx_etapadeproceso_displayname = selected_features.cmtx_etapadeproceso_displayname.str.replace("ó",'o')
#    etapa = pd.DataFrame({'etapa': selected_features.cmtx_etapadeproceso_displayname})    
#    selected_features = pd.concat([selected_features, pd.get_dummies(etapa, prefix='etapa')],axis=1)
    selected_features.drop(['cmtx_etapadeproceso_displayname'],axis=1, inplace=True)

    aux = []
    for value in selected_features.cmtx_commitedtoforecast:
        if value == 'No':
            aux.append(0)
        if value == 'Sí':
            aux.append(1)
    selected_features['cmtx_commitedtoforecast'] = pd.DataFrame(aux, dtype=int)    
    
    ##### One-Hot encode para negocios relevantes o no relevantes.
    aux = []
    for value in selected_features.totalamount_base:
        if value > 100000:
            aux.append(1)
        else:
            aux.append(0)
            
    relevante = pd.DataFrame({'relevante': aux})
    selected_features = pd.concat([selected_features, relevante],axis=1)
    
    
    #### One-Hot encode para numero de dias

    aux_quarter = []
    aux_semester = []
    aux_3quarter = []
    aux_year = []
    aux_overyear = []
    for value in aux:
        if(value <= 90):
            aux_quarter.append(1) 
            aux_semester.append(0)
            aux_3quarter.append(0)
            aux_year.append(0) 
            aux_overyear.append(0) 
        elif(value > 90 and value <= 180):
            aux_quarter.append(0) 
            aux_semester.append(1)
            aux_3quarter.append(0)
            aux_year.append(0) 
            aux_overyear.append(0) 
        
        elif(value < 180 and value <= 270):
            aux_quarter.append(0) 
            aux_semester.append(0)
            aux_3quarter.append(1)
            aux_year.append(0) 
            aux_overyear.append(0) 
            
        elif(value > 270 and value <= 360):
            aux_quarter.append(0) 
            aux_semester.append(0)
            aux_3quarter.append(0)
            aux_year.append(1) 
            aux_overyear.append(0) 
            
        elif(value > 360):
            aux_quarter.append(0) 
            aux_semester.append(0)
            aux_3quarter.append(0)
            aux_year.append(0) 
            aux_overyear.append(1) 
            
    aux = [aux_quarter, aux_semester, aux_3quarter, aux_year, aux_overyear]
    
    periodOfTime = pd.DataFrame({'period_1quarter': aux[0], 'period_2quarter': aux[1], 'period_3quarter': aux[2], 'period_4quarter': aux[3], 'period_overayear': aux[4]})
    selected_features = pd.concat([selected_features, periodOfTime],axis=1)
    
    
    
    #### Ganada o perdida? #####

    
    
    #Ordenar por fecha de cierre
    selected_features = selected_features.set_index(['cmtx_fechacierreoportunidad'])
    selected_features.sort_index(inplace=True)
    
    # Separamos en dos archivos las cargas de filiales en crm y fuera de crm.
    cargas = ['MX - NOVIS','QUINTEC CHILE DISTRIB', 'CL - WIRELESS', 'CL - TECNOGLOBAL', 'CL - SOLEX', 'CL - SERVIBANCA', 'CL - NOVIS', 'CL - MICROGEO', ' CL - EDUCACIÓN', 'BR - PARS', 'BR - CTIS', 'BR - ATIVAS']


    ## Cargas de filiales.
    fr = []
    for i in range(0,len(cargas)):
        fr.append(selected_features.loc[selected_features['cmtx_unidaddenegocioname'] == cargas[i]])
    #definir dataframe con filiales fuera de CRM
    filial_nocrm = pd.concat(fr)

    for i in range(0,len(cargas)):
        selected_features = selected_features[selected_features['cmtx_unidaddenegocioname'] != cargas[i]]
    selected_features.drop(['cmtx_filialname'],axis=1, inplace=True)
    #el resto queda como filial dentro de CRM
    selected_features = selected_features.dropna()
    processed_features = selected_features 
    
    return processed_features
    

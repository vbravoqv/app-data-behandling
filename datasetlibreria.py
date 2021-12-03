#Se importan librerias
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly import tools
import plotly.offline as py
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#Clase encargada de almacenar el dataframe cargado desde el uploader
class update_data:
    #constructor:
    def __init__(self,data_file):
        self.data_file=data_file

    #Metodo que crea el dataframe y lo almacena en una variable  
    def createdataframe(self):
        data_file=self.data_file
        if data_file is not None:
            st.write("Nombre del archivo: "+data_file.name)            
            data=pd.read_csv(data_file)
            st.dataframe(data)
            return data
        else:
            pass 

    #Metodo que crea un dataframe filtrado por las columnas elegidas por el usuario
    def filter(self,df):
        data_file=self.data_file
        if data_file is not None:
            if st.checkbox('Filtar columnas del dataset'):
                st.sidebar.subheader('Filtado de columnas')
                data_cols_to_show=st.sidebar.multiselect('Data set',df.columns)
                st.write("Data set filtrado")
                st.write(df[data_cols_to_show])
                datafilter=df[data_cols_to_show]
                return datafilter
            else:
                 return df
#clase con diferentes metodos para comenzar el procesamiento de data            
class process_data:       
    #constructor:
    def __init__(self,dffilter):
        self.dffilter=dffilter

    #Creación del boton, generar perfil de datos.
    def buttomprofile(self):
        dffilter=self.dffilter
        if dffilter is not None:
            if st.button("Obtener perfil de datos"):
                totalcolumnas=resumendata(dffilter)
                typedata(dffilter,totalcolumnas)
                nulls(dffilter)
                describe(dffilter)
                correlation(dffilter)
                df=fillnanobject(dffilter)
                df=fillnancolumns(df)
                df=dataprocess(df)
                outliners(df)
            elif dffilter is not None:
                totalcolumnas=resumendata(dffilter)
                typedata(dffilter,totalcolumnas)
                nulls(dffilter)
                describe(dffilter)
                correlation(dffilter)
                df=fillnanobject(dffilter)
                df=fillnancolumns(df)
                df=dataprocess(df)
                outliners(df)
        else:
            st.write("Ingreso al else")


#funciones para procesamiento de datos

#Función que permite obtener el total de filas y columnas de un dataframe
def resumendata(dffilter):
    st.title("Perfil de datos")
    dfrowsandcolumns=dffilter.shape
    st.markdown("---")
    st.write("Resumen")
    m1, m2, m3 = st.columns((1,1,1))
    st.markdown("---")
    m1.write('')
    m2.metric(label ='Total filas',value=dfrowsandcolumns[0])
    m3.metric(label ='Total Columnas',value=dfrowsandcolumns[1])
    m1.write('')
    return dfrowsandcolumns[1]

#Función que permite obtener el total de filas y columnas de un dataframe
def createdataframe(df,columna1,columna2):
    df=df.to_frame().reset_index()
    columns=df.columns
    df = df.rename(columns= {columns[0]: columna1,columns[1]:columna2})
    return df

#Función que permite obtener el total de filas por tipo de dato
def typedata(dffilter,totalcolumnas):
    totalcolumnas=totalcolumnas
    dfdtypes=dffilter.dtypes
    dfdtypes=createdataframe(dfdtypes,"Column","Type")
    dfdtypecount=dfdtypes['Type'].value_counts()
    dfdtypecount=createdataframe(dfdtypecount,"Type","Total")
    long=len(dfdtypecount)
    with st.container():
        st.write("Tipos de datos")
        m=st.columns([1]*int(long))
        #st.markdown("---")
        for i in dfdtypecount.index:  
            m[i].metric(label=str(dfdtypecount["Type"][i]),value=str(dfdtypecount["Total"][i]))
            with st.expander(str(dfdtypecount["Type"][i])):
                st.write(str(round((int(dfdtypecount["Total"][i])/int(totalcolumnas))*100))+"%")
                st.progress (round((int(dfdtypecount["Total"][i])/int(totalcolumnas))*100))
                st.write(dffilter.select_dtypes(dfdtypecount["Type"][i]).columns)         
        with st.expander("Vista general"):
            st.table(dfdtypes.astype(str))
        st.markdown("---")

#Función que permite obtener el total de datos nulos por columnas
def nulls(dffilter):
    dfrowsandcolumns=dffilter.shape
    valuerows=dfrowsandcolumns[0]
    valuecolumns=dfrowsandcolumns[1]
    totaldata=int(valuerows*valuecolumns)
    nullssumtotal= dffilter.isnull().sum().sum()
    dfnullssum=dffilter.isnull().sum()
    dfnullssum=createdataframe(dfnullssum,"Columns","Nulls")
    graph=px.bar(dfnullssum,dfnullssum['Nulls'],dfnullssum['Columns'],color=dfnullssum['Columns'])
    m1,m2=st.columns([3,1])
    with m2:
        st.metric(label ='Total datos invalidos',value=int(nullssumtotal),delta_color='inverse')
    with m1:    
        st.write("Calidad de los datos")
        st.write('Datos validos: '+str(100-round((nullssumtotal/totaldata)*100))+"%")
        st.write('Datos invalidos: '+str(round((nullssumtotal/totaldata)*100))+"%")
        st.progress(100-round((nullssumtotal/totaldata)*100))
    with st.expander("Ver grafica"):
        st.plotly_chart(graph)  
        #st.plotly_chart(graph)  
            

    with st.expander("Ver detalle"):
        dfnullssum=dfnullssum.sort_values('Nulls',ascending=False)
        for i in dfnullssum.index:
            st.write(str(dfnullssum["Columns"][i]))
            st.write("Valores faltantes : "+str(dfnullssum["Nulls"][i]))
            st.write(str(round((int(dfnullssum["Nulls"][i])/int(valuerows))*100))+"%")
            st.progress (round((int(dfnullssum["Nulls"][i])/int(valuerows))*100))
    st.markdown("---")


#Función que permite obtener una descripción de los datos
def describe(dffilter):
    st.write("Descripción")
    selectcolumns=st.selectbox('',dffilter.columns, key=1)
    st.write("Dispersión por columna")
    plot = px.histogram(dffilter,x=selectcolumns) 
    st.plotly_chart(plot)
    generaldescripcion=dffilter.describe()
    with st.expander("Ver descripción completa"):
        st.write(generaldescripcion)
    st.markdown("---")

#Función que permite obtener la correlación de los datos
def correlation(dffilter):
    st.write("Matriz de correlación")
    correlation_mat=dffilter.corr()
    corr_pairs=correlation_mat.unstack()
    sorted_pairs = corr_pairs.sort_values(kind="quicksort")
    negative_pairs = sorted_pairs[sorted_pairs < 0]
    strong_pairs = sorted_pairs[sorted_pairs > 0.5]
    with st.expander("Pares con correlación negativa"):
        st.write(negative_pairs)
    with st.expander("Pares con correlación fuerte"):
        st.write(strong_pairs)
    p=sns.heatmap(correlation_mat,linewidths=.5)
    st.pyplot(p.figure)
    st.markdown("---")



#Función que remplaza los nan de las columnas tipo object
def fillnanobject(dffilter):
    for col in dffilter:
        if dffilter[col].dtypes == 'object':
            dffilter[col] = dffilter[col].fillna(dffilter[col].mode().iloc[0])

    return dffilter     

#Función que remplaza los nan de las columnas float
def fillnancolumns(dffilter):
   dffilter=dffilter.fillna(dffilter.mean())
   return dffilter     

#Función que convierte al mejor tipo de dato posible 
def dataprocess(dffilter):
    dfn = dffilter.convert_dtypes()
    return dfn

#Función que entrega los valores atipicos 
def outliners(dffilter):
    totalcolumns=int(dffilter.shape[0])
    listcolumns=list(dffilter.columns.values)
    dffilterfreq = dffilter.groupby(listcolumns).size().reset_index(name='n') 
    dffilterfreq['support'] = dffilterfreq['n']/dffilterfreq.shape[0]
    dffilterfreq_values=dffilterfreq[["n","support"]]


    #Parametrización del modelo
    #number=st.select_slider("Seleccione el tresh",value=(-0,5, 0,5))
    number =st.number_input('Ingrese el umbral de datos atipicos')
    iforest = IsolationForest(n_estimators=100, max_samples='auto', 
                          contamination=0.05, max_features=1.0, 
                          bootstrap=False, n_jobs=-1, random_state=1)


    iforest.fit(dffilterfreq_values)
    scores = iforest.decision_function(dffilterfreq_values)
    dffilterfreq['scores']=scores
    dffilterfreq_values['scores']=scores
    thresh = np.quantile(dffilterfreq_values.scores, number)


    dffilterfreq['anomaly']=np.where(dffilterfreq_values['scores']< thresh,1,0)
    dfanomaly=dffilterfreq[dffilterfreq['anomaly'] == 1]
    
    m1,m2=st.columns([2,1])
    anomalysum=int(dfanomaly.shape[0])
    with m2:
        m1.metric(label='Total datos atipicos',value=str(anomalysum))
        
    with m1:
        st.write("Porcentaje de datos atipicos")
        st.write('Datos atipicos: '+str(round((anomalysum/totalcolumns)*100))+"%")
        division=anomalysum/totalcolumns
        st.progress(round((anomalysum/totalcolumns)*100))
        
    st.write(dfanomaly)
   
    
     


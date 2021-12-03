#se importan librerias
import pandas as pd
import streamlit as st
import numpy as np
from datasetlibreria import update_data,process_data
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Estructura de la aplicación web
st.title("Data-behandling")
menu=["Home","Cargar Dataset","Conjuntos de datos"]
choice=st.sidebar.selectbox("Menu",menu)

#Se validan elecciones de selectbox   
if choice=="Home":
    st.subheader("Home")

if choice=="Cargar Dataset":
   data_file=st.file_uploader("Upload CSV",type=["csv"]) 
   data=update_data(data_file)
   df=data.createdataframe() 
   datafilter=data.filter(df) 
   buttom= process_data(datafilter)
   #col1,col2=st.columns((2,3))
   with st.container():
    buttom.buttomprofile()
    

    #update_data.createdataframe
    



# """ def main():
#     st.title("File Upload")

#     menu=["Home","Cargar Dataset","Conjuntos de datos"]
#     choice=st.sidebar.selectbox("Menu",menu)

#     if choice=="Home":
#         st.subheader("Home")

#     elif choice=="Cargar Dataset":
#         st.subheader("Dataset")
#         data_file=st.file_uploader("Upload CSV",type=["csv"])
#         if data_file is not None:
#             #st.write(type(data_file))
#             #file_details={"Nombre del archivo":data_file.name,"Tipo de archivo":data_file.type,"Tamaño":data_file.size}
#             st.write("Nombre del archivo: "+data_file.name)
#             data=pd.read_csv(data_file)
#             st.dataframe(data)
#             if st.checkbox('Filtar columnas del dataset'):
#                st.sidebar.subheader('Filtado de columnas')
#                data_cols_to_show=st.sidebar.multiselect('Data set',data.columns)
#                st.write("Data set filtrado")
#                st.write(data[data_cols_to_show])

#         if st.button("Obtener perfil de datos"):
#             dfrowsandcolumns=data.shape
#             st.title("Perfil de datos")
#             m1, m2, m3 = st.columns((1,1,1))
#             #Total filas y columnas
#             st.markdown("---")
#             m1.write('')
#             m2.metric(label ='Total filas',value=dfrowsandcolumns[0])
#             m3.metric(label ='Total Columnas',value=dfrowsandcolumns[1])
#             m1.write('')
#             #Cantidad de columnas con un tipo de dato determinado
#             g1, g2, g3 = st.columns((1,1,1))
#             dfdtypes=data.dtypes
#             dfdtypes = dfdtypes.to_frame().reset_index()
#             dfdtypes = dfdtypes.rename(columns= {'index': 'Column',0:'Type'})
#             st.write("Columnas")
#             g1.write(dfdtypes.astype(str))
#             dfdtypepercent=(dfdtypes['Type'].value_counts()/dfdtypes['Type'].count())*100
#             dfdtypepercent = dfdtypepercent.to_frame().reset_index()
#             #porcentaje de columnas con un tipo de dato determinado
#             dfdtypepercent= dfdtypepercent.rename(columns= {'index': 'Type','Type':'Total'})
#             g2.write(dfdtypepercent.astype(str))
#             #cantidad de columnas con un tipo de dato determinado
#             dfdtypecount=dfdtypes['Type'].value_counts()
#             dfdtypecount = dfdtypecount.to_frame().reset_index()
#             dfdtypecount = dfdtypecount.rename(columns= {'index':'Type','Type':'Total'})
#             st.write(dfdtypecount.astype(str))  
#             #Número total de valores perdidos por columna
#             dfnullssum=data.isnull().sum()
#             dfnullssum = dfnullssum.to_frame().reset_index()
#             dfnullssum = dfnullssum.rename(columns= {'index':'Column',0:'nulls'})
#             st.write(dfnullssum.astype(str)) 
#             #Número total de valores perdidos en todo el dataframe
#             nullssumtotal= data.isnull().sum().sum()
#             st.write('Total datos nulos en el dataset: '+nullssumtotal.astype(str)) 


#     elif choice=="Conjuntos de datos":
#         st.subheader("Conjuntos de datos")
    
#     else:
#         st.subheader("About")

# if __name__ =='__main__':
#     main() 




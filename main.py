
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
import streamlit as st
from src.test import unique_test , pass_video, delete_files_in_folder
import os
import base64
import os
import time


import zipfile


UPLOAD_FOLDER_videos = 'videos'
UPLOAD_FOLDER_model = 'models'
test_species_dic = { 'Panthera onca' : 0,  'Puma concolor': 1,  'Leopardus pardalis' : 2,
                     'Crax rubra' : 3,  'Aramides albiventris' : 4, 'Aramus Guarauna' : 5}
@st.cache_resource
def save_uploaded_file(uploaded_file, path):
    # Crea la carpeta de subida si no existe
    if not os.path.exists(path):
        os.makedirs(path)

    # Guarda el archivo en la carpeta de subida
    file_path = os.path.join(path, uploaded_file.name)
    with open(file_path, 'wb') as file:
        file.write(uploaded_file.getbuffer())

    return file_path


# Función para guardar el DataFrame como archivo CSV
@st.cache_resource
def guardar_como_csv(dataframe):
    csv_file = "resultados.csv"
    dataframe.to_csv(csv_file, index=False)
    return csv_file
#load model 

@st.cache_resource
def get_download_link(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".csv":
        with open(file_path, "r") as file:
            csv_string = file.read()
        base64_encoded = base64.b64encode(csv_string.encode()).decode("utf-8")
        href = f'<a href="data:file/csv;base64,{base64_encoded}" download="{file_path}">Download CSV</a>'
    elif file_extension == ".avi":
        with open(file_path, "rb") as file:
            zip_data = file.read()
        base64_encoded = base64.b64encode(zip_data).decode("utf-8")
        href = f'<a href="data:application/zip;base64,{base64_encoded}" download="{file_path}">Download ZIP</a>'
    else:
        href = ""

    return href

@st.cache_resource
def zip_videos(folder_path, zip_path,video_file):
    current_path = os.getcwd()
    folder_path = "".join([current_path, folder_path])
    # Obtener la ruta del archivo en el sistema de archivos
    file_path = "".join([folder_path, video_file.name])
    # Escribir el archivo en el zip
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_path, os.path.basename(file_path))

#model = YOLO('../models/model_l.pt')  # load model
test_species_dic = {  0 : 'Panthera onca' ,  1: 'Puma concolor',  2 : 'Leopardus pardalis' , 3 : 'Crax rubra' ,  4 : 'Aramides albiventris' , 5 : 'Aramus Guarauna' }

@st.cache_resource
def load_model(model_file,UPLOAD_FOLDER_model) :
    file_path = save_uploaded_file(model_file,UPLOAD_FOLDER_model)
    model = YOLO(file_path)  # Cargar el modelo desde el archivo
    st.write("Model loaded successfully.")
    return model

def change_extension(file_name):
    parts = file_name.split('.')
    if parts[-1] != '.avi':
        name = parts[0] + '.avi'
        return name
    else:
        return file_name

def get_download_link(file_path):
    with open(file_path, "rb") as file:
        file_content = file.read()
    return file_content

def main():
    st.title('Species model')
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Configurar la carga de archivos
    st.set_option('deprecation.showfileUploaderEncoding', False)

    model_file = st.file_uploader("Upload Model", type=["pt"])

    if model_file is not None:
      model = load_model(model_file,UPLOAD_FOLDER_model)
      
    uploaded_files = st.file_uploader("Load CSV", accept_multiple_files=True)

    download_csv = st.checkbox("Download CSV")
    download_videos = st.checkbox("Download Videos")

    if st.button("Predict"):
        df_con = pd.DataFrame()
        prediccion = []

        if uploaded_files is not None:
            for video_file in uploaded_files:

                # Procesar el video y generar el CSV
                print(video_file)
                file_path = save_uploaded_file(video_file,UPLOAD_FOLDER_videos)
                df_final = unique_test(model,file_path, spe = None , dic =  test_species_dic , change = True)
                df_final = df_final[df_final['%'] == df_final['%'].max()]
                df_con = pd.concat([df_con,df_final], axis = 0)

                download_clicked = st.button("Download")
                if download_videos:
                    pass_video()
                    if download_clicked :

                    
                        video_path = os.listdir( 'datasets/')
                        st.write("Download Videos:")
                        file_content = get_download_link('datasets/' + video_path[0])
                        #st.markdown(get_download_link('datasets/' + video_path[0]), unsafe_allow_html=True)
                        st.download_button(label="Download CSV", data=file_content, file_name=file_path)
                        #delete_files_in_folder('/datasets/')
                    if not download_clicked:
                        st.markdown("Click the button to download the file.")
            delete_files_in_folder('/videos/')
            
            csv_file = guardar_como_csv(df_con)



            # Mostrar el enlace de descarga para el archivo CSV
            if download_csv : 
                st.write("download  CSV:")
                st.markdown(get_download_link(csv_file), unsafe_allow_html=True)
            

            pass_video()



        time.sleep(15)
        delete_files_in_folder('/datasets/')

      
    if st.button("Clear All"):
        # Clears all st.cache_resource caches:
        st.cache_resource.clear()
if __name__ == '__main__' :
    main()

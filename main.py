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


UPLOAD_FOLDER = 'videos'
test_species_dic = { 'Panthera onca' : 0,  'Puma concolor': 1,  'Leopardus pardalis' : 2,
                     'Crax rubra' : 3,  'Aramides albiventris' : 4, 'Aramus Guarauna' : 5}
def save_uploaded_file(uploaded_file):
    # Crea la carpeta de subida si no existe
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Guarda el archivo en la carpeta de subida
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, 'wb') as file:
        file.write(uploaded_file.getbuffer())

    return file_path

# Función para guardar el DataFrame como archivo CSV
def guardar_como_csv(dataframe):
    csv_file = "resultados.csv"
    dataframe.to_csv(csv_file, index=False)
    return csv_file
#load model 


def get_download_link(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".csv":
        with open(file_path, "r") as file:
            csv_string = file.read()
        base64_encoded = base64.b64encode(csv_string.encode()).decode("utf-8")
        href = f'<a href="data:file/csv;base64,{base64_encoded}" download="{file_path}">Download CSV</a>'
    elif file_extension == ".zip":
        with open(file_path, "rb") as file:
            zip_data = file.read()
        base64_encoded = base64.b64encode(zip_data).decode("utf-8")
        href = f'<a href="data:application/zip;base64,{base64_encoded}" download="{file_path}">Download ZIP</a>'
    else:
        href = ""

    return href


def zip_videos(folder_path, zip_path):
    current_path = os.getcwd()
    # Check if the folder path exists
    folder_path = "".join([current_path, folder_path])
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.basename(file_path))

model = YOLO('../models/model_l.pt')  # load model
test_species_dic = {  0 : 'Panthera onca' ,  1: 'Puma concolor',  2 : 'Leopardus pardalis' , 3 : 'Crax rubra' ,  4 : 'Aramides albiventris' , 5 : 'Aramus Guarauna' }



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
                file_path = save_uploaded_file(video_file)
                df_final = unique_test(model,file_path, spe = None , dic =  test_species_dic , change = True)
                df_final = df_final[df_final['%'] == df_final['%'].max()]
                df_con = pd.concat([df_con,df_final], axis = 0)

            delete_files_in_folder('/videos/')
            
            csv_file = guardar_como_csv(df_con)

            # Mostrar el enlace de descarga para el archivo CSV
            st.write("download  CSV:")
            st.markdown(get_download_link(csv_file), unsafe_allow_html=True)
            

        pass_video()

        if download_videos:
            zip_path = "videos.zip"
            zip_videos('/datasets/', zip_path)
            st.write("Download Videos:")
            st.markdown(get_download_link(zip_path), unsafe_allow_html=True)

        time.sleep(15)
        delete_files_in_folder('/datasets/')
if __name__ == '__main__' :
    main()
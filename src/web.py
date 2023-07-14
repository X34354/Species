from ultralytics import YOLO
import pandas as pd
import numpy as np
import streamlit as st
from test import unique_test , list_test , pass_video, delete_files_in_folder
import os
import base64
import os
import time
import shutil
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
# Función para generar el enlace de descarga para el archivo CSV
def get_download_link(file_path):
    with open(file_path, "r") as file:
        csv_string = file.read()
    base64_encoded = base64.b64encode(csv_string.encode()).decode("utf-8")
    href = f'<a href="data:file/csv;base64,{base64_encoded}" download="{file_path}">Descargar</a>'
    return href

@st.cache_data(show_spinner=False)
def get_files_in_folder(carpeta):
    return os.listdir(carpeta)

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

def descargar_videos():
    ruta_carpeta = r'C:/Users/fraja/Documents/pista/codigo/src/datasets'
    archivos = os.listdir(ruta_carpeta)
    
    for archivo in archivos:
        if archivo.endswith('.mp4', '.avi'):
            ruta_archivo = os.path.join(ruta_carpeta, archivo)
            st.write(f"Descargando archivo: {archivo}")
            # Aquí puedes agregar la lógica para descargar el archivo
        
    st.write("Descarga completa")


def descargar_archivos_carpeta(carpeta):
    # Obtiene la lista de archivos en la carpeta
    archivos = os.listdir(carpeta)

    # Muestra los nombres de los archivos en la interfaz de Streamlit
    st.write(f"Archivos en la carpeta '{carpeta}':")
    for archivo in archivos:
        st.write(archivo)

    # Agrega un enlace de descarga para cada archivo
    st.write("Descargar archivos:")
    for archivo in archivos:
        ruta_archivo = os.path.join(carpeta, archivo)
        with open(ruta_archivo, "rb") as f:
            contenido = f.read()
        contenido_base64 = base64.b64encode(contenido).decode("utf-8")
        enlace_descarga = f'<a href="data:file/txt;base64,{contenido_base64}" download="{archivo}">{archivo}</a>'
        st.markdown(enlace_descarga, unsafe_allow_html=True)

def zip_videos(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.basename(file_path))

model = YOLO('G:/model_l.pt')  # load model
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
                
                #results =  model.predict (video_file , save = False, conf=0.65,  show = False )
                # Mostrar el resultado del CSV
                # Guardar el DataFrame como archivo CSV
            delete_files_in_folder('C:/Users/fraja/Documents/pista/codigo/src/videos/')
            
            csv_file = guardar_como_csv(df_con)

            # Mostrar el enlace de descarga para el archivo CSV
            st.write("download  CSV:")
            st.markdown(get_download_link(csv_file), unsafe_allow_html=True)
            

        pass_video()

        if download_videos:
            pass_video()
            zip_path = "videos.zip"
            zip_videos('C:/Users/fraja/Documents/pista/codigo/src/datasets/', zip_path)
            st.write("Download Videos:")
            st.markdown(get_download_link(zip_path), unsafe_allow_html=True)
        delete_files_in_folder('C:/Users/fraja/Documents/pista/codigo/src/datasets/')
if __name__ == '__main__' :
    main()
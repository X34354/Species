import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
import streamlit as st
from src.test import unique_test , pass_video
import os
import base64
import os
import time 

directory_path = "datasets/" 
directory_path_video = "videos/"  
directory_path_csv = 'csvs folder/'
inactivity_time = 3600  
UPLOAD_FOLDER_videos = 'videos'
UPLOAD_FOLDER_model = 'models'

test_species_dic = {  0 : 'Panthera onca' ,  1: 'Puma concolor',  2 : 'Leopardus pardalis' ,
                     3 : 'Crax rubra' ,  4 : 'Aramides albiventris' , 5 : 'Aramus Guarauna' }

def clear_cache():
    st.caching.clear_cache()

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

def delete_files_in_directory(directory):
    # Get the list of files in the specified directory
    files = os.listdir(directory)

    # Iterate through each file and delete it
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

    print("Deletion of files completed.")

@st.cache_resource
def guardar_como_csv(dataframe):
    csv_file = "csvs folder/resultados.csv"
    dataframe.to_csv(csv_file, index=False)
    return csv_file

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

def hide_streamlit_menu_footer():
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
def process_uploaded_files(model, uploaded_files):
    df_con = pd.DataFrame()

    for video_file in uploaded_files:
        file_path = save_uploaded_file(video_file, UPLOAD_FOLDER_videos)
        df_final = unique_test(model, file_path, spe=None, dic=test_species_dic, change=False)
        df_final = df_final[df_final['%'] == df_final['%'].max()]
        df_con = pd.concat([df_con, df_final])

    return df_con

def download_files(directory, message, file_extension):
    csv_path = os.listdir(directory)
    st.write(message)
    for ele_csv in csv_path:
        if ele_csv.endswith(file_extension):
            file_content = get_download_link(os.path.join(directory, ele_csv))
            st.download_button(label=str(ele_csv), data=file_content, file_name=ele_csv)
def delete_inactive_files(directory_path, inactivity_time):
    # Get the list of files in the specified directory
    files = os.listdir(directory_path)

    for file in files:
        # Get the full path of the file
        file_path = os.path.join(directory_path, file)

        # Check if the file has been modified within the last 5 minutes
        if time.time() - os.path.getmtime(file_path) > inactivity_time:
            # Delete the file
            os.remove(file_path)
            print(f"File deleted: {file_path}")

def check_create_folder(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created folder: {directory_path}")
    else:
        print(f"Folder already exists: {directory_path}")            
def main():

    check_create_folder(directory_path)
    check_create_folder(directory_path_video)
    check_create_folder(directory_path_csv)

    st.title('Species model')
    hide_streamlit_menu_footer()
    st.set_option('deprecation.showfileUploaderEncoding', False)

    model_file = st.file_uploader("Upload Model", type=["pt"])

    if model_file is not None:
        model = load_model(model_file, UPLOAD_FOLDER_model)

    uploaded_files = st.file_uploader("Load CSV", accept_multiple_files=True)
    download_csv = st.checkbox("Download CSV")
    download_videos = st.checkbox("Download Videos")


    if st.button("Predict"):
        if uploaded_files is not None:
            df_con = process_uploaded_files(model, uploaded_files)
            _ = guardar_como_csv(df_con)
            pass_video()
            
    if download_csv and (len( os.listdir(directory_path_csv)) != 0):
        download_files(directory_path_csv, "Download CSV:" , '.csv')

    if download_videos and (len( os.listdir(directory_path_video)) != 0) :

        download_files(directory_path, "Download VIDEOS:" , '.avi')

    if st.button("delete all (videos , csvs and cache)"):
        delete_files_in_directory(directory_path)
        delete_files_in_directory(directory_path_video)
        delete_files_in_directory(directory_path_csv)
        st.cache_resource.clear()

    delete_inactive_files(directory_path, inactivity_time)
    delete_inactive_files(directory_path_video, inactivity_time)
    delete_inactive_files(directory_path_csv, inactivity_time)
    
if __name__ == '__main__' :
    main()

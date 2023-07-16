from ultralytics import YOLO
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import shutil
import json



def plot_confusion_matrix(confusion_matrix_result, test_species_dic):
    """
    Plots the confusion matrix.

    Args:
        confusion_matrix_result (numpy.ndarray): Confusion matrix.
        test_species_dic (dict): Dictionary mapping species labels to their corresponding values.
    """
    species_names = list(test_species_dic.values())
    tick_labels = [f'{i}\n{species_names[i]}' for i in range(len(species_names))]
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_result, annot=True, fmt="d", cmap="Blues",
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def evaluate_classification(prediccion, labels, test_species_dic, plot_matrix=False):
    """
    Evaluates a classification model by calculating the confusion matrix, precision, and F1 score based on
    predicted values and true labels. Optionally, it can plot the confusion matrix.

    Args:
        prediccion (list): List of predicted species labels.
        labels (list): List of true species labels.
        test_species_dic (dict): Dictionary mapping species labels to their corresponding values.
        plot_matrix (bool, optional): Whether to plot the confusion matrix. Defaults to False.

    Returns:
        dict: A dictionary containing confusion matrix, precision, and F1 score.
    """
    # Replace predicted and true labels with their corresponding values from the dictionary

    prediccion = [test_species_dic.get(elemento, elemento) for elemento in prediccion]

    labels = [test_species_dic.get(elemento, elemento) for elemento in labels]

    # Calculate confusion matrix

    confusion_matrix_result = confusion_matrix(labels, prediccion)
    
    # Calculate precision and F1 score
    precision = precision_score(labels, prediccion, average='macro')
    f1 = f1_score(labels, prediccion, average='macro')

    # Create a dictionary to store the results
    results = {
        'confusion_matrix': confusion_matrix_result,
        'precision': precision,
        'f1_score': f1
    }

    # Plot confusion matrix if requested
    if plot_matrix:
        plot_confusion_matrix(confusion_matrix_result, test_species_dic)

    return results

def extract_result( results ) :
    df = pd.DataFrame()
    l = []
    c = []
    for result in results:

        prueba = result.boxes  
        if len(prueba.cls) == 0 :
            l.append('no detections')
            c.append(np.nan)
        else :
            l = prueba.cls.tolist() + l
            c = prueba.conf.tolist() + c
    df['prediccion'] = l 
    df['conf'] = c
    return df 

def unique_test(model , name_video : str ,spe  = None, dic = None , change = False) :
    df_final = pd.DataFrame() 
    results =  model.predict (name_video , save = True, conf=0.65,  show = False , project ='datasets' , name = None , stream = True, imgsz=224)

    df =  extract_result( results )

    df_f = df[df['prediccion'] != 'no detections']
    df_f = df_f['prediccion'].value_counts().to_frame()

    for ind in list(df_f.index) :

        contener_df = pd.DataFrame() 
        df_filter = df_f[df_f.index == ind]
        contener_df.loc[0, '%'] = df_filter['count'].values[0] / df.shape[0]
        contener_df.loc[0, 'mean'] = np.around(df['conf'][ df['prediccion'] == ind ].mean(), 3)
        contener_df.loc[0, 'Label'] = df_f.index[0]
        df_final = pd.concat([df_final,contener_df],axis = 0)

    df_final['File'] = name_video
    if change :
        df_final['Label'] = df_final['Label'].map(dic)
    if spe is None : 
        return df_final
    else :
        df_final['Specie'] = spe
    
        
    return df_final


def list_test(model ,listt : list  , df_val  : list  , df) :
    df_con = pd.DataFrame()
    labels = []
    prediccion = []

    i = 0
    for ele ,spe in zip(listt , df_val)  :
        df_final = unique_test(model , ele  , spe ) 

        df_con = pd.concat([df_con,df_final], axis = 0)
        if len(df_final) == 0 : 
            prediccion.append(6)
        else : 
            df_final = df_final[df_final['%'] == df_final['%'].max()]
            prediccion.append(df_final['Specie'].iloc[0])

        labels.append(df['SpeciesID'].iloc[i])
        i += 1
    return df_con , labels , prediccion



def delete_files_in_folder(folder_path):
    """
    Deletes all files within a folder.
    """
    #folder_path = 'C:/Users/fraja/Documents/pista/codigo/src/videos/'  # Replace with the actual folder path
    current_path = os.getcwd()
    # Check if the folder path exists
    folder_path = "".join([current_path, folder_path])
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Invalid folder path: {folder_path}")
        return

    # Iterate over the files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check if the current item is a file
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except OSError as e:
                print(f"Error deleting file: {file_path}\n{e}")

def pass_video():
    # Main directory path
    main_directory = '/datasets'
    current_path = os.getcwd()

    # Check if the main directory path exists
    main_directory = "".join([current_path, main_directory])

    # Get the list of folders in the main directory
    folders = next(os.walk(main_directory))[1]

    # Iterate over each folder
    for folder in folders:
        # Get the full path of the folder
        folder_path = os.path.join(main_directory, folder)

        # Get the list of files in the folder
        files = os.listdir(folder_path)

        # Iterate over each file in the folder
        for file in files:
            # Get the full path of the file
            file_path = os.path.join(folder_path, file)

            # Check if the file already exists in the destination folder
            destination_file_path = os.path.join(main_directory, file)
            count = 1
            while os.path.exists(destination_file_path):
                # Append a number to the file name for the copy
                file_name, extension = os.path.splitext(file)
                new_file_name = f"{file_name}_copy{count}{extension}"
                destination_file_path = os.path.join(main_directory, new_file_name)
                count += 1

            # Move the file to the destination folder
            shutil.move(file_path, destination_file_path)

        # Remove the empty folder
        os.rmdir(folder_path)

if __name__ == '__main__' :

    data = open('../jsons/path_general.json')
    data = json.load(data)

    path = data['Videos']
    p = data['Val'] + '/Crax rubra/Sar_1_25.AVI'
    test_species_dic = { 'Panthera onca' : 0,  'Puma concolor': 1,  'Leopardus pardalis' : 2, 'Crax rubra' : 3,  'Aramides albiventris' : 4, 'Aramus Guarauna' : 5}
    test_species_dic = {  0 : 'Panthera onca' ,  1: 'Puma concolor',  2 : 'Leopardus pardalis' , 3 : 'Crax rubra' ,  4 : 'Aramides albiventris' , 5 : 'Aramus Guarauna' }

    df_val = pd.read_csv(data['Data_filter'])
    df_val  = df_val[df_val['Ml'] == 'val']
    df_val = df_val[df_val['SpeciesID'].isin(test_species_dic.keys())]
    df_val['File'] = df_val['File'].apply(lambda x: path + x)
    n = 10
    
    df_val = df_val.groupby('SpeciesID').head(10)

    model = YOLO( data['directory'] + 'model_l.pt')  # load model

    df_final = unique_test(model , p , 'Crax', test_species_dic , change = True)
    df, labels , prediccion = list_test(model ,list(df_val['File']) , list(df_val['SpeciesID']) , df_val) 

    results = evaluate_classification(prediccion, labels, test_species_dic)




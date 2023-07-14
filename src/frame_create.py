import os
import pandas as pd
import cv2
import random

def extract_random_frames(video_path, frame_count, output_folder, name):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(total_frames * 0.02)  # Excluir los primeros 2 segundos
    end_frame = int(total_frames * 0.98)  # Excluir los últimos 2 segundos

    frames_to_extract = min(frame_count, end_frame - start_frame)
    extracted_frames = 0
    current_frame = 0

    random_frame_indices = random.sample(range(start_frame, end_frame), frames_to_extract)

    frames = []
    while cap.isOpened() and extracted_frames < frames_to_extract:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in random_frame_indices:
            frames.append(frame)
            extracted_frames += 1
            # Guardar el frame en la carpeta de destino
            frame_filename = f"{name}_{current_frame}.jpg"
            
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)

        current_frame += 1

    cap.release()

    return frames


def extract_frames(video_path, frame_count, output_folder,name):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(total_frames * 0.02)  # Excluir los primeros 2 segundos
    end_frame = int(total_frames * 0.98)  # Excluir los últimos 2 segundos

    frames_to_extract = min(frame_count, end_frame - start_frame)
    extracted_frames = 0
    current_frame = 0

    frames = []
    while cap.isOpened() and extracted_frames < frames_to_extract:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame >= start_frame and current_frame <= end_frame:
            frames.append(frame)
            extracted_frames += 1
            # Guardar el frame en la carpeta de destino
            frame_filename = f"{name}_{extracted_frames}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)

        current_frame += 1

    cap.release()

    return frames

def extract_frames_from_videos(df, frame_count, output_folder):
    total_videos = len(df)
    if frame_count // total_videos == 0 :
        df = df.sample(n=frame_count)
    frames_per_video = frame_count // total_videos
    remaining_frames = frame_count % total_videos

    frames = []
    for _, row in df.iterrows():
        video_path = row['File']

        file_name = os.path.basename(video_path)
        name_without_extension = os.path.splitext(file_name)[0]

        # Remove additional characters
        name = name_without_extension.replace("_", " ")

        if remaining_frames > 0:
            video_frames = extract_random_frames(video_path, frames_per_video + 1, output_folder,name)
            remaining_frames -= 1
        else:
            video_frames = extract_random_frames(video_path, frames_per_video, output_folder,name)

        frames.extend(video_frames)

    return frames


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")


if __name__ == '__main__' :
    
    l = ['Train', 'Test', 'Val']
    create_folder("G:/data/Train")
    create_folder("G:/data/Test")
    create_folder("G:/data/Val")
    

    df = pd.read_csv('../data/df_filter.csv')
    df_specie = pd.read_excel('../data/Especies Red.xlsx')
    # Iterate over the names in the 'name' column of the DataFrame
    for nom in l : 
        for name in df_specie['SpeciesID']:
            name = 'G:/data/' + nom + '/' + name
            create_folder(name)
    

    frame = ['train', 'test']
    for name in df_specie['SpeciesID']: 
        print('specie: ', name)
        for dataset in frame  : 
            if dataset == 'train' :
                frame_count = 400
            else :
                frame_count = 130
            output_folder = 'G:/data/' + str(dataset) + '/' + str(name) + '/' 
            df_filter_dataset = df[df['SpeciesID'] == name]
            df_filter_dataset = df_filter_dataset[df_filter_dataset['Ml'] == dataset]
            df_filter_dataset['File'] = df_filter_dataset['File'].apply(lambda x: 'G:/videos/' + x)
            extracted_frames = extract_frames_from_videos(df_filter_dataset, frame_count, output_folder)

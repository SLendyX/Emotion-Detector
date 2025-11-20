import os
import cv2
import numpy
from keras.utils import to_categorical

#----Configurare
#Cai catre date
BASE_DIR = "data"
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

#Dimensiune imagine
IMG_SIZE = 48

#Categoriile de emotii
CATEGORIES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def load_and_preprocess_data(data_type):
    #data_type: "train" sau "test"

    path = os.path.join(RAW_DIR, data_type)

    images_list = []
    labels_list = []

    print(f"Începem procesarea pentru setul: {data_type}...")

    for category in CATEGORIES:
        folder_path = os.path.join(path, category)
        class_num = CATEGORIES.index(category)


        if not os.path.exists(folder_path):
            print(f"Atenție: Folderul {folder_path} nu există. Sărim peste.")
            continue

        for img_name in os.listdir(folder_path):
            try:
                #Citirea imaginii
                img_path = os.path.join(folder_path, img_name)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img_array is None:
                    continue

                #Redimensionare
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                


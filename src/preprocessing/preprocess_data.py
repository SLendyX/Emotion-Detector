import os
import cv2
import numpy as np
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

                #Adaugam imaginea si eticheta in lista
                images_list.append(resized_array)
                labels_list.append(class_num)

            except Exception as e:
                print(f"Eroare la imagine {img_name}: {e}")

    #Conversia in Numpy Arrays
    x = np.array(images_list)
    y = np.array(labels_list)

    #Normalizarea datelor
    x = x/255.0

    #Reshape pentru CNN
    # Rețeaua așteaptă 4 dimensiuni: (nr_poze, inaltime, latime, nr_canale)
    # Noi avem (nr_poze, 48, 48) -> Trebuie să devină (nr_poze, 48, 48, 1)
    x = x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    #One-Hot Encoding pentru etichete           
    y = to_categorical(y, num_classes=len(CATEGORIES))

    print(f"   Gata {data_type}! Am procesat {len(x)} imagini.")
    print(f"   Forma datelor x: {x.shape}")
    print(f"   Forma etichetelor y: {y.shape}")
    
    return x, y
    
if __name__ == "__main__":
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    x_train, y_train = load_and_preprocess_data("train")

    x_test, y_test = load_and_preprocess_data("test")

    print(" Salvarea fișierelor .npy...")
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), x_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), x_test)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

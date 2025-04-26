# 1. Import bibliotek
import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import kagglehub

"""# Download latest version
path = kagglehub.dataset_download("robinreni/signature-verification-dataset")

print("Path to dataset files:", path)"""


# 2. Wstępne przetwarzanie obrazu
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = 255 - binary  # Inwersja jeśli tło jest białe
    # Ścienianie
    skeleton = cv2.ximgproc.thinning(binary)
    return skeleton


# 3. Ekstrakcja cech
def extract_features(image):
    # Przykładowe cechy: rozmiar podpisu, liczba czarnych pikseli, aspekt ratio
    height, width = image.shape
    black_pixels = np.sum(image == 0)
    aspect_ratio = width / height
    return [black_pixels, aspect_ratio]


# 4. Wczytaj dane
def load_data(dataset_path):
    features = []
    labels = []
    for user_folder in os.listdir(dataset_path):
        user_path = os.path.join(dataset_path, user_folder)
        if os.path.isdir(user_path):
            for img_name in os.listdir(user_path):
                img_path = os.path.join(user_path, img_name)
                img = preprocess_image(img_path)
                feat = extract_features(img)
                features.append(feat)
                labels.append(user_folder)
    return np.array(features), np.array(labels)


# 5. Klasyfikacja
def train_and_evaluate(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")


# 6. Główne wywołanie
if __name__ == "__main__":
    dataset_path = "path_to_dataset"  # <- tutaj folder ze zdjęciami
    features, labels = load_data(dataset_path)
    train_and_evaluate(features, labels)

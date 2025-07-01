import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import matplotlib
import os

if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")


# Punkt e: Implementacja klasyfikatora k-NN oraz MLP
def train_and_evaluate(features, labels, model_name="multi"):
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../outputs", exist_ok=True)

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42)

    scaler = StandardScaler()  # normalizacja cech
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Klasyfikator k-NN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)
    knn_acc = accuracy_score(y_test, knn.predict(X_test_scaled)) * 100

    # Klasyfikator MLP (sieć neuronowa)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=2000, random_state=42, alpha=0.001, early_stopping=True)
    mlp.fit(X_train_scaled, y_train)
    mlp_acc = accuracy_score(y_test, mlp.predict(X_test_scaled)) * 100

    # Random Forest
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test_scaled)) * 100

    # SVM
    svm = SVC(class_weight='balanced')
    svm.fit(X_train_scaled, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test_scaled)) * 100

    # Raporty
    print(f"\n[{model_name}] Dokładność k-NN: {knn_acc:.2f}%")
    print(f"[{model_name}] Dokładność MLP:  {mlp_acc:.2f}%")
    print(f"[{model_name}] Dokładność RF:   {rf_acc:.2f}%")
    print(f"[{model_name}] Dokładność SVM:  {svm_acc:.2f}%\n")

    y_pred = mlp.predict(X_test_scaled)
    print(f"Raport MLP - {model_name}:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Macierz pomyłek
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Macierz pomyłek MLP - {model_name}")
    plt.xlabel("Przewidziane")
    plt.ylabel("Rzeczywiste")
    plt.savefig(f"outputs/confusion_matrix_{model_name}.png")
    plt.close()

    # Zapis modelu i skalera
    joblib.dump(mlp, f"models/mlp_model_{model_name}.pkl")
    joblib.dump(knn, f"models/knn_model_{model_name}.pkl")
    joblib.dump(rf, f"models/rf_model_{model_name}.pkl")
    joblib.dump(svm, f"models/svm_model_{model_name}.pkl")
    joblib.dump(scaler, f"models/scaler_{model_name}.pkl")
    joblib.dump(le, f"models/label_encoder_{model_name}.pkl")
    print(f"Zapisano modele i encoder do: models/mlp_model_{model_name}.pkl, models/scaler_{model_name}.pkl, models/label_encoder_{model_name}.pkl")

## 🖊 Signature Recognition and Verification System
![Tests](https://github.com/wojtekjurkowicz/signature_recognition/actions/workflows/tests.yml/badge.svg)

This project implements a full pipeline for offline signature recognition and verification using classical machine learning methods. It includes custom image preprocessing, handcrafted feature extraction, training of classification models (k-NN, MLP, RandomForest, SVM), and visual analysis of feature space using PCA and t-SNE.

---

### 🔧 Features

* **Custom Preprocessing**: Binarization, segmentation, and thinning of signature images.
* **Handcrafted Feature Extraction**: Extracts shape, geometry, and statistical features (e.g., Hu Moments, pixel distribution).
* **CSV Export**: All extracted features and labels are saved to `outputs/features.csv`.
* **Classification Models**:

  * Multi-class classification (identifying the signer)
  * Binary classification (genuine vs. forged signatures)
  * Models used: k-NN, MLP, RandomForest, SVM
* **Balanced Training**: Class imbalance is mitigated using `class_weight='balanced'` for applicable models.
* **Visualization Tools**: Dimensionality reduction with PCA and t-SNE, plus confusion matrices.
* **Dataset Loader**: Automatically downloads and processes the Kaggle signature dataset.

---

### 📁 Project Structure

```
.
├── scripts/
│   ├── main.py                    # Main pipeline: load, process, classify
│   ├── models.py                  # Model training and evaluation
│   └── visualization.py           # Feature space visualization
```

---

### 🚀 How to Run

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

Minimal requirements:

```txt
numpy
opencv-python
scikit-learn
matplotlib
seaborn
kagglehub
Pillow
```

2. **Run the main script**:

```bash
python scripts/main.py
```

This will:

* Download the dataset via KaggleHub
* Preprocess images and extract features
* Save features to `outputs/features.csv`
* Train k-NN, MLP, RandomForest, and SVM classifiers
* Generate evaluation reports and confusion matrices
* Create PCA and t-SNE visualizations

---

### 🧪 Run Tests

To run unit tests:

```bash
pytest -v tests/
```
Make sure you're in the root folder (signature_recognition/) and that dependencies are installed.

---

### 🧪 Model Output

* `*_model_*.pkl`: Trained models (MLP, RF, etc.)
* `scaler_*.pkl`, `label_encoder_*.pkl`: Used for normalization and label encoding
* `confusion_matrix_*.png`: Confusion matrices for each classification task
* `pca_visualization.png`, `tsne_visualization.png`: Feature space visualizations
* `features.csv`: Extracted feature vectors with labels (used for experiments and analysis)

---

### 📦 Dataset

Dataset is fetched from [Kaggle](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset) using the `kagglehub` API. It includes genuine and forged signatures from multiple users.

---

### 🧑‍💻 Author

Project developed for educational and experimental purposes. Ideal for exploring traditional signature verification techniques without deep learning.
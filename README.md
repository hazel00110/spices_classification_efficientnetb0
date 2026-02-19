# 🌿 Indonesian Spices Image Classification using EfficientNetB0

A deep learning project that classifies 4 types of Indonesian spices (rempah) from images using transfer learning with EfficientNetB0.

---

## 📌 Project Overview

This project builds an image classification model to identify common Indonesian spices:

| Class | Indonesian Name | English Name |
| ----- | --------------- | ------------ |
| 0     | Jahe            | Ginger       |
| 1     | Kencur          | Sand Ginger  |
| 2     | Kunyit          | Turmeric     |
| 3     | Lengkuas        | Galangal     |

The model leverages **EfficientNetB0** pretrained on ImageNet as a feature extractor, with custom classification layers on top.

---

## 📁 Dataset

- **Source:** [https://drive.google.com/file/d/1Fhi6-6V4kKdxgiplUt38iVlolN6tCYHd/view?usp=sharing](https://drive.google.com/file/d/1Fhi6-6V4kKdxgiplUt38iVlolN6tCYHd/view?usp=sharing)
- **Structure:** One folder per class, containing `.jpg`, `.jpeg`, and `.webp` images
- **Split:** 80% training / 20% testing (with 20% of training used for validation)

---

## 🏗️ Model Architecture

```
Input (224x224x3)
    ↓
Data Augmentation (RandomFlip, RandomRotation, RandomZoom, RandomContrast)
    ↓
EfficientNetB0 (pretrained, frozen — ImageNet weights)
    ↓
Dense(128, ReLU) → Dropout(0.45)
    ↓
Dense(256, ReLU) → Dropout(0.45)
    ↓
Dense(4, Softmax)
```

---

## ⚙️ Training Configuration

| Parameter      | Value                                      |
| -------------- | ------------------------------------------ |
| Input Size     | 224 × 224                                  |
| Batch Size     | 32                                         |
| Optimizer      | Adam (lr=0.0001)                           |
| Loss           | Categorical Crossentropy                   |
| Max Epochs     | 100                                        |
| Early Stopping | patience=5 (val_loss)                      |
| LR Scheduler   | ReduceLROnPlateau (factor=0.2, patience=3) |

---

## 📊 Results

The model was evaluated on the held-out test set. Metrics reported include accuracy, precision, recall, F1-score per class, and a confusion matrix.

> Training/validation accuracy and loss curves are visualized in the notebook.

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- EfficientNetB0 (Transfer Learning)
- scikit-learn
- NumPy, Pandas
- Matplotlib

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/spices-classification.git
cd spices-classification
```

### 2. Install dependencies

```bash
pip install tensorflow keras scikit-learn pandas numpy matplotlib seaborn pillow
```

### 3. Download the dataset

Download the dataset from [https://drive.google.com/file/d/1Fhi6-6V4kKdxgiplUt38iVlolN6tCYHd/view?usp=sharing](https://drive.google.com/file/d/1Fhi6-6V4kKdxgiplUt38iVlolN6tCYHd/view?usp=sharing) and place it in a folder named `Rempah_Dataset/` with the following structure:

```
Rempah_Dataset/
├── jahe/
├── kencur/
├── kunyit/
└── lengkuas/
```

### 4. Run the notebook

```bash
jupyter notebook spices_classification.ipynb
```

---

## 📂 Project Structure

```
spices-classification/
├── spices_classification.ipynb   # Main notebook
├── Rempah_Dataset/               # Dataset directory (not included)
└── README.md
```

---

## 👤 Author

**Hazel Pernanda Putra**

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

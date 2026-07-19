# Handwritten Digit Recognition using Artificial Neural Network (ANN)

A deep learning project that recognizes handwritten digits (0–9) using an Artificial Neural Network (ANN) built with PyTorch and trained on the MNIST dataset.

> **Note:** The trained model (`.pth`) and preprocessing artifact (`.pkl`) files are **not included** in this repository to keep the repository lightweight. These files can be generated locally by training the model.

---

## 📌 Project Overview

This project classifies handwritten digits (0–9) using an Artificial Neural Network (ANN). The model is trained on the MNIST dataset and can be integrated with a web application for real-time handwritten digit recognition.

---

## 🚀 Features

- Handwritten digit recognition (0–9)
- Built with PyTorch
- Custom MNIST Dataset Loader
- ANN architecture with multiple hidden layers
- Batch Normalization
- Dropout Regularization
- Model evaluation using Accuracy and Classification Report
- Model artifact generation
- FastAPI backend support
- Easy integration with web frontend

---

## 🧠 Model Architecture

```
Input Image (28 × 28)

        │

Flatten (784)

        │

Linear (784 → 512)

        │

Batch Normalization

        │

ReLU

        │

Dropout (0.3)

        │

Linear (512 → 256)

        │

Batch Normalization

        │

ReLU

        │

Dropout (0.3)

        │

Linear (256 → 128)

        │

Batch Normalization

        │

ReLU

        │

Linear (128 → 10)

        │

Output (Digit 0–9)
```

---

## 📊 Dataset

**Dataset:** MNIST Handwritten Digits

| Item | Value |
|------|------|
| Training Images | 60,000 |
| Testing Images | 10,000 |
| Image Size | 28 × 28 |
| Color Mode | Grayscale |
| Classes | 10 (Digits 0–9) |

---

## ⚙️ Technologies Used

- Python
- PyTorch
- NumPy
- Scikit-learn
- Joblib
- FastAPI
- HTML
- CSS
- JavaScript

---

## 🏋️ Model Training

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Epochs | 20 (configurable) |

---

## 📈 Evaluation Metrics

The trained model is evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Classification Report

Example output:

```
Accuracy: 98.45%
```

---

## 📦 Model Artifacts

The following files are **generated after training** and are **not included** in this repository.

```
digit_ann_model.pth
digit_pipeline.pkl
model_metadata.pkl
```

---

## 🎯 Future Improvements

- CNN implementation for higher accuracy
- Real-time webcam digit recognition
- Mobile-friendly UI
- Docker support
- Cloud deployment
- ONNX model export
- Performance optimization

---

## 👨‍💻 Author

**Dinuka Samarakoon**

AI Engineering Student

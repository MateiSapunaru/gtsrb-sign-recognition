# 🚦 Traffic Sign Recognition with Explainable AI

An end-to-end Computer Vision system for traffic sign classification using deep learning, enhanced with explainability techniques and deployed as an interactive web application.

---

## 📌 Overview

This project implements a complete machine learning pipeline for traffic sign recognition using the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset. The system leverages **transfer learning**, **model interpretability (Grad-CAM)**, and **real-time inference** through a **Streamlit-based application**.

The goal is not only to achieve high classification performance but also to provide **transparent and interpretable predictions**, making the system suitable for real-world applications such as autonomous driving and driver assistance systems.

---

## 🧠 Key Features

* ✅ Transfer Learning using EfficientNet
* ✅ Two-stage training (frozen backbone + fine-tuning)
* ✅ High-performance classification (≈98% test accuracy)
* ✅ Grad-CAM visual explanations
* ✅ Batch explainability analysis (correct vs incorrect predictions)
* ✅ Interactive Streamlit UI:

  * Image upload inference
  * Webcam snapshot inference
  * Real-time webcam prediction (WebRTC)
* ✅ Fully containerized using Docker

---

## 🏗️ System Architecture

The project follows a modular and production-oriented structure:

```text
src/
├── data/              # Data preprocessing and loaders
├── models/            # Training, evaluation, inference
├── explainability/    # Grad-CAM implementation
├── serving/           # Inference logic for app
app/
├── pages/             # Streamlit UI pages
configs/               # Config-driven pipeline
artifacts/             # Models, metrics, predictions
```

The pipeline is fully **config-driven**, enabling reproducibility and easy experimentation.

---

## 📊 Model Performance

The model was evaluated on a **held-out test set** to ensure unbiased performance estimation.

### 🔢 Final Metrics

* **Test Accuracy:** ~98.27%
* **Macro F1 Score:** ~0.97
* **Weighted F1 Score:** ~0.98

These results indicate strong generalization, with minimal discrepancy between training, validation, and test performance.

---

📸 screenshot no.1 (metrics, etc) here

---

## 📈 Evaluation Insights

* The confusion matrix shows a **strong diagonal**, indicating high classification accuracy across most classes.
* Misclassifications are concentrated in:

  * visually similar classes (e.g., speed limits)
  * low-quality or low-resolution images
* Slight performance degradation is observed in underrepresented or ambiguous classes such as:

  * pedestrians
  * road condition signs (e.g., ice/snow, bumpy road)

---

📸 screenshot no.2 (confusion matrix) here

---

## 🔍 Explainability with Grad-CAM

To ensure interpretability, Grad-CAM is used to visualize the regions of the image that influence the model’s predictions.

### Observations:

* The model consistently focuses on:

  * the sign shape (circular/triangular)
  * internal symbols (numbers, icons)
* Incorrect predictions often correlate with:

  * low-resolution inputs
  * occlusion or motion blur
  * ambiguous visual patterns

This confirms that the model relies on **meaningful features**, not background artifacts.

---

📸 screenshot no.3 (grad-cam examples) here

---

## 🎥 Real-Time Inference

The application supports **live webcam inference** using WebRTC:

* Frame-by-frame classification
* Efficient inference (runs every N frames for performance)
* Real-time overlay of predictions

This demonstrates the system’s ability to operate in **near real-time environments**.

---

📸 screenshot no.4 (live webcam) here

---

## 🧪 Challenges & Design Considerations

### 1. Class Imbalance

The dataset contains uneven distribution across classes, requiring:

* careful evaluation (macro vs weighted metrics)
* analysis of per-class accuracy

---

### 2. Visual Ambiguity

Some classes are inherently difficult due to:

* low contrast symbols
* similar geometric shapes
* small object size within the image

This leads to confusion between semantically similar categories.

---

### 3. Resolution Constraints

Traffic signs are often small in images, which:

* limits fine-grained feature extraction
* reduces Grad-CAM spatial precision

---

### 4. Real-Time Performance

Balancing inference speed with accuracy required:

* reducing inference frequency (frame skipping)
* avoiding heavy explainability (Grad-CAM) in live mode

---

### 5. Generalization vs Dataset Bias

While validation accuracy was extremely high, true performance was assessed using a separate test set to ensure:

* no data leakage
* realistic generalization behavior

---

## 🐳 Deployment

The application is fully containerized using Docker.

### Run locally:

```bash
docker compose build
docker compose up
```

Then open:

```text
http://localhost:8501
```

---

## 🛠️ Tech Stack

* **Python**
* **PyTorch**
* **Torchvision**
* **OpenCV**
* **Streamlit**
* **streamlit-webrtc**
* **Grad-CAM**
* **Docker**
* **Poetry**

---

## 🚀 Future Improvements

* Model ensembling (EfficientNet + ResNet)
* Data augmentation for hard classes
* Grad-CAM++ for finer localization
* Mobile deployment (ONNX / TensorRT)
* Edge deployment for embedded systems

---

## 📌 Conclusion

This project demonstrates a complete machine learning lifecycle:

* data processing
* model training
* evaluation
* explainability
* deployment

The system achieves high accuracy while maintaining interpretability and real-time usability, making it a strong candidate for practical applications in intelligent transportation systems.

---

# Traffic Sign Recognition with Explainable AI and ESP32 Streaming

An end-to-end computer vision system for traffic sign classification using deep learning, enhanced with explainability techniques and deployed with a real ESP32-CAM streaming pipeline.

---

## Overview

This project implements a complete machine learning pipeline for traffic sign recognition using the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset.

The system combines:

* deep learning (PyTorch, EfficientNet)
* explainability (Grad-CAM)
* real-time inference (Streamlit)
* embedded systems (ESP32-CAM streaming)

The objective is not only high accuracy, but also building a **realistic edge AI system** that integrates hardware and software components.

---

## Key Features

* Transfer learning using EfficientNet
* Two-stage training (frozen backbone + fine-tuning)
* High-performance classification (~98% accuracy)
* Grad-CAM visual explanations
* Per-class evaluation and confusion matrix
* Real-time inference from ESP32 camera stream
* Interactive Streamlit dashboard
* Dockerized deployment

---

## System Architecture

```text
ESP32 Camera (C++)
        ↓
HTTP MJPEG Stream
        ↓
OpenCV (Python)
        ↓
PyTorch Model (Inference)
        ↓
Streamlit UI
```

Project structure:

```text
src/
├── data/
├── models/
├── explainability/
├── serving/
app/
├── pages/
configs/
artifacts/
```

---

## Model Performance

Evaluation was performed on a held-out test set:

* Accuracy: 98.27%
* Macro F1 Score: 0.97
* Weighted F1 Score: 0.98

The model shows strong generalization and consistent performance across most classes.

---

📸 <img width="2000" height="1000" alt="overall_test_metrics" src="https://github.com/user-attachments/assets/7ed4e779-e8ac-4319-a2b9-ba88906baf1b" />
<img width="2400" height="1200" alt="training_validation_accuracy" src="https://github.com/user-attachments/assets/ff1c20ba-6b89-40c1-ac25-7e855d73081c" />
<img width="2400" height="1200" alt="training_validation_loss" src="https://github.com/user-attachments/assets/859d5fb5-8ce5-421b-ace7-0f84a8299faf" />


---

## Evaluation Insights

* The confusion matrix shows a strong diagonal (high accuracy)
* Most errors occur between visually similar classes
* Lower performance appears in:

  * pedestrians
  * bumpy road
  * ice/snow signs

---

📸 <img width="3000" height="2786" alt="confusion_matrix" src="https://github.com/user-attachments/assets/fe43d330-0093-4332-9caf-2f30cdd59142" />


---

📸 <img width="3180" height="1579" alt="per_class_accuracy" src="https://github.com/user-attachments/assets/24fac041-16d2-4493-b26d-91c12cf7982a" />

<img width="2800" height="1200" alt="lowest_performing_classes" src="https://github.com/user-attachments/assets/828bcabf-f976-46cb-9b35-1efe2ea29cae" />


---

## Explainability with Grad-CAM

Grad-CAM is used to visualize the regions influencing predictions.

Observations:

* The model focuses on:

  * sign shape (circle, triangle)
  * internal symbols and numbers
* Errors correlate with:

  * blur
  * occlusion
  * low resolution

This confirms the model relies on meaningful visual features.

---

📸 <img width="1600" height="722" alt="image" src="https://github.com/user-attachments/assets/d8998e04-0a42-4587-b17c-11d0ea478832" />


---

## ESP32-CAM Integration

The system uses an ESP32-CAM module to simulate a real-world edge vision setup.

### How it works

1. ESP32 captures frames using its onboard camera
2. Streams video via HTTP (MJPEG)
3. Streamlit connects using OpenCV
4. Frames are processed by the model
5. Predictions are displayed in real time

---

### Stream URL

```
http://<ESP32-IP>:81/stream
```

Example:

```
http://192.168.1.144:81/stream
```

---

### Notes

* Only one client can access the stream at a time
* Close browser camera preview before using Streamlit
* ESP32 and PC must be on the same network

---

📸 <img width="1181" height="1118" alt="image" src="https://github.com/user-attachments/assets/2aded4da-ad01-4398-99f4-31a6cf5f4a92" />


---

## Real-Time Inference

* Live MJPEG stream processing
* Frame-by-frame classification
* Adjustable inference frequency
* Real-time overlay of predictions

Performance is optimized by running inference every N frames and avoiding heavy operations during streaming.

---

## Electronics and Hardware

### ESP32-CAM

A compact microcontroller with:

* WiFi connectivity
* OV2640 camera sensor
* onboard JPEG encoding
* HTTP streaming capability

📸 <img width="2048" height="1536" alt="image" src="https://github.com/user-attachments/assets/5f069d61-c50b-456e-9fc4-f114fe5434ef" />

---

### Required Components

* ESP32-CAM module
* FTDI programmer (for flashing)
* 5V power supply
* jumper wires

---

### Hardware Flow

```text
Camera Sensor
      ↓
ESP32 Microcontroller
      ↓ (WiFi stream)
PC (Streamlit + Model)
      ↓
Prediction Output
```

---

### Limitations

* limited onboard processing power
* dependent on WiFi stability
* occasional frame drops

---

## ESP32 Firmware (C++)

The ESP32 runs a modified version of the **CameraWebServer** example.

Responsibilities:

* initialize the camera
* connect to WiFi
* start HTTP server
* stream MJPEG frames

Key concepts:

```cpp
esp_camera_init(&config);
WiFi.begin(ssid, password);
```

The stream is exposed at:

```cpp
"/stream"
```

Frames are continuously captured and sent over HTTP.

Important:
The ESP32 does not run the machine learning model. It acts only as a streaming device, while inference is performed on the PC.

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

---

### 2. Install Dependencies

```bash
poetry install
poetry shell
```

---

### 3. Run the Application

```bash
streamlit run app/main.py
```

Open:

```
http://localhost:8501
```

---

### 4. Run ESP32 Stream (Optional)

1. Flash the ESP32-CAM with `CameraWebServer.ino`
2. Connect it to WiFi
3. Get its IP address

Set stream URL:

```
http://<ESP32-IP>:81/stream
```

Then open the **ESP32 Stream** page in the app and start streaming.

---

### Troubleshooting

* Ensure ESP32 and PC are on the same network
* Close other clients using the stream
* If OpenCV fails, ensure FFMPEG support is installed
* Adjust inference frequency for better performance

---

### Docker (Alternative)

```bash
docker compose build
docker compose up
```

Then open:

```
http://localhost:8501
```

---

## Tech Stack

* Python
* PyTorch
* OpenCV
* Streamlit
* ESP32 (C++)
* Grad-CAM
* Docker
* Poetry

---

## Future Improvements

* Edge inference (ONNX / TensorRT)
* Model optimization for embedded devices
* Improved data augmentation for difficult classes
* Grad-CAM++ for finer explanations

---

## Conclusion

This project demonstrates a complete **end-to-end AI system**, combining:

* data processing
* model training and evaluation
* explainability
* real-time deployment
* embedded hardware integration

It reflects a realistic pipeline applicable to intelligent transportation and edge AI systems.

---

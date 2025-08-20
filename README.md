# Automated Defect Detection and Visual Inspection for Quality Control in Manufacturing

## 📌 Project Overview

This project implements an **automated defect detection framework** for manufacturing quality control using deep learning. Traditional manual inspection is slow, inconsistent, and error-prone. To address this, we leverage **Convolutional Autoencoder (CAE)**, **U-Net**, and **PatchCore** to detect, classify, and localize defects in industrial products.

The system is built on the **MVTec Anomaly Detection (AD) dataset**, a benchmark dataset with 5,000+ high-resolution images covering 15 categories (e.g., bottle, cable, wood, hazelnut, leather). Each category contains **normal images, defective images, and pixel-level ground truth masks**.

---

## 🎯 Objectives

* Develop a **deep learning–based defect detection system** (CAE, U-Net, PatchCore).
* Perform **Exploratory Data Analysis (EDA)** on the MVTec AD dataset.
* Visualize **class distribution, defect patterns, and anomaly heatmaps**.
* Train models for **classification, segmentation, and anomaly detection**.
* Build an **interactive Gradio UI** for real-time inspection.

---

## 📂 Dataset

* **Name**: MVTec Anomaly Detection Dataset (MVTec AD)
* **Link**: [MVTec AD Official Website](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
* **Categories**: 15 (objects + textures)
* **Labels**: `good` (non-defective) vs. `defective`
* **Defects**: scratches, contamination, cracks, holes, misalignment, etc.
* **Format**: `.png` images, structured by category

---

## ⚙️ Methodology

1. **Data Preprocessing**

   * Image resizing, normalization
   * Augmentation (flipping, rotation, cropping)
   * Train/test split

2. **Exploratory Data Analysis (EDA)**

   * Class distribution plots
   * Sample image visualization (good vs defective)
   * Defect overlay with ground-truth masks
   * Heatmaps for defect-prone regions

3. **Model Development**

   * **CAE** → Detect global anomalies via reconstruction error
   * **U-Net** → Pixel-level segmentation for defect localization
   * **PatchCore** → Feature memory–based detection of subtle anomalies

4. **Evaluation Metrics**

   * CAE → Mean Squared Error (MSE)
   * U-Net → Mean anomaly mask probability, IoU
   * PatchCore → Feature-space anomaly score

---

## 🛠️ Tools & Technologies

* **Programming**: Python
* **Deep Learning**: PyTorch, TensorFlow/Keras
* **Image Processing**: OpenCV, Albumentations
* **Data Handling**: NumPy, Pandas
* **Visualization**: Matplotlib, Seaborn, Plotly, Grad-CAM
* **Deployment**: Gradio UI for interactive visualization
* **Environment**: Google Colab (GPU-enabled)

---

## 📊 Results

From evaluation on test images:

* **CAE** → Best at identifying *global anomalies*, but less sensitive to localized defects.
* **U-Net** → Strong at *localizing spatially extensive anomalies* with segmentation masks.
* **PatchCore** → Most sensitive to *subtle feature-level deviations*.

✅ **Multi-model integration provides balanced detection capability.**

---

## 🚀 Expected Outcomes

* Reliable defect detection with **>90% target accuracy**.
* Automated inspection reducing manual effort and errors.
* Visual insights (heatmaps, anomaly overlays) for better interpretability.
* Scalable solution adaptable to multiple industries (electronics, automotive, packaging, etc.).

---

## 📉 Limitations & Future Work

* Model generalization may be limited for unseen defect types.
* Real-world deployment may face challenges (lighting variation, occlusion).
* Currently static images → future work includes **real-time video inspection**.
* Edge deployment for **on-site inference** in factories.
* Possible extension to **multi-class detection (YOLO, Mask R-CNN, etc.)**.

---

## 🖥️ Running the Project

### Requirements

```bash
pip install torch torchvision opencv-python albumentations matplotlib seaborn plotly gradio
```

### Training

* Mount dataset in Google Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

* Run training script (`CAE`, `U-Net`, `PatchCore`) → models saved in `mvtec_bottle_models/`.

### Gradio UI

Launch interactive defect detection:

```python
import gradio as gr
# Run app
demo.launch()
```



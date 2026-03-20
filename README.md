# 🧠 Brain Tumor MRI Image Classification

## 📌 Project Overview
This project builds a deep learning model to classify brain MRI images into tumor categories using both a custom CNN and transfer learning models.

The goal is to assist in automated detection of brain tumors from MRI scans.

---

## 🧩 Dataset
The dataset consists of MRI images categorized into:

- Glioma
- Meningioma
- Pituitary
- No Tumor

### Folder Structure
data/
│
├── train/
├── valid/
└── test/

---

## ⚙️ Workflow

1. Dataset Loading  
2. Data Preprocessing (Normalization, Resizing)  
3. Data Augmentation  
4. Custom CNN Model  
5. Transfer Learning Models  
6. Model Training  
7. Model Evaluation  
8. Model Comparison  
9. Streamlit Deployment  

---

## 🤖 Models Used

### Custom Model
- Convolutional Neural Network (CNN)

### Transfer Learning Models
- ResNet50  
- MobileNetV2  
- InceptionV3  
- EfficientNetB0  

---

## 📊 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-Score  

---

## 🏆 Best Model Selection
All models are evaluated on the test dataset, and the best-performing model is selected based on accuracy.

---

## 🚀 Streamlit Application

The project includes an interactive web app where users can:

- Upload MRI images  
- Get predicted tumor type  
- View confidence scores 
---

## ▶️ Run the Application

streamlit run app.py

Open in browser:  
http://localhost:8501

---

## 📁 Project Structure
.
├── data/
├── models/
├── app.py
├── notebook.ipynb
├── requirements.txt
└── README.md

---

## ⚠️ Disclaimer
This project is for educational purposes only and should not be used for medical diagnosis.

---

## 📌 Future Improvements
- Improve accuracy with fine-tuning  
- Add Grad-CAM visualization  
- Deploy on cloud (AWS / GCP / Azure)  

---

## 👨‍💻 Author
Kalki Prathisha K


## UI Snapshot:

<img width="573" height="840" alt="image" src="https://github.com/user-attachments/assets/07733a71-4004-41c4-b9a0-633f92800524" />

<img width="657" height="809" alt="image" src="https://github.com/user-attachments/assets/1016f0d7-7b41-4737-8a4b-209e1ce8886e" />


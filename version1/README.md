# 🧠 NeuroVision AI – Version 1  
### Brain Tumor Detection using Deep Learning

---

## 📌 Overview
**NeuroVision AI (Version 1)** is a medical imaging system designed to detect brain tumors  (Deep Learning).

The system analyzes brain MRI images and provides:
- Tumor classification  
- Probability scores  
- Visual explanation using heatmaps  

It also includes a user-friendly interface built with **Gradio**, making it easy for users to interact with the model.

---

## 🎯 Key Features
- 🧠 Brain tumor detection using deep learning  
- 📊 High accuracy: **98.5%**  
- 🔥 Heatmap visualization (Grad-CAM style)  
- 🖥️ Interactive UI using Gradio  
- 📈 Probability display for predictions  
- 🩺 Basic medical recommendations based on patient age  
- 👁️ Clear visualization of input and output  

---

## 🧰 Technologies Used
- Python 3.x  
- PyTorch  
- Torchvision  
- Gradio  
- NumPy  
- Matplotlib  
- PIL  

---

## 🖥️ User Interface
The system uses **Gradio** to provide a simple and interactive interface:

- Upload MRI image  
- View prediction instantly  
- See probability distribution  
- Visualize heatmap highlighting important regions  

---

## 🤖 Model
- Built using **PyTorch**  
- Trained on brain MRI image dataset  
- Optimized for classification accuracy  
- Achieved **98.5% accuracy**  

---

## 🔥 Heatmap Visualization
The system generates a heatmap to highlight important regions in the brain image that influenced the model's prediction.

This improves:
- Model interpretability  
- Trust in predictions  
- Medical relevance  

---

## 🩺 Medical Guidance
The system provides basic medical suggestions based on:
- Patient age  
- Prediction results  

> ⚠️ Note: This system is for educational purposes and does not replace professional medical diagnosis.

---

## 📁 Project Structure

version1/
│
├── notebooks/
│ └── training_v1.ipynb
│
├── models/
│ └── model_v1.pth
│
├── data/
│ └── sample_images/
│
├── results/
│ ├── predictions/
│ └── heatmaps/
│
├── src/
│ └── app.py # Gradio interface
│
└── README.md


---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

2. Run the application
python src/main.py

📊 Results
High classification accuracy
Clear visualization of predictions
Reliable heatmap explanations
⚠️ Disclaimer

This project is intended for:

Educational purposes
Research and experimentation

It is not a substitute for medical diagnosis.

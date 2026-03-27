# 🧠 NeuroVision AI  
**AI-Powered Medical Imaging System for Brain Tumor & Stroke Detection**

---

## 📌 Overview
**NeuroVision AI** is a cutting-edge medical imaging project leveraging deep learning to detect brain abnormalities, including tumors and strokes, using CT and MRI scans.  

The system evolves in **two major versions**, showing a progression from a simple baseline model to a full multi-model intelligent solution.  

**Key Focus Areas:**  
- Brain tumor detection  
- Stroke detection  
- Cross-modality analysis (CT vs MRI)  
- Explainable AI using heatmaps  
- Multi-language medical recommendations  

---

## 🚀 Project Versions

### 🔹 Version 1 – Baseline System
- Detects **brain tumors only**  
- **Single deep learning model**  
- Limited dataset  
- Basic Gradio interface  
- Accuracy: **98.5%**  
- Provides **heatmap visualization** highlighting tumor location  
- Offers **basic medical recommendations** based on patient age  

### 🔹 Version 2 – Advanced System
- Detects **brain tumors and strokes**  
- **Multi-model architecture** (4 models) specialized by image type (CT/MRI) and condition (tumor/stroke)  
- Supports **CT and MRI scans**  
- Expanded dataset for improved accuracy  
- Simultaneous prediction using two models for cross-modality analysis  
- Enhanced **heatmap visualization** for better interpretability  
- Multi-language interface (**English/Arabic**)  
- Personalized medical advice with **intensity adjustment** based on severity  
- Provides external resources and guidance for patients  

---

## 📊 Detailed Version Comparison

| Feature | Version 1 | Version 2 |
|--------|----------|----------|
| Diseases | Tumor | Tumor + Stroke |
| Models | 1 | 4 |
| Dataset | Small | Large |
| Accuracy | 98.5% | Higher |
| Imaging Modalities | Limited | CT + MRI |
| User Interface | Basic | Advanced Gradio app |
| Language Support | Single | English + Arabic |
| Heatmap | Basic | Enhanced with gradation |
| Recommendations | Simple | Personalized + intensity adjustment |
| Cross-Modality Analysis | ❌ | ✅ |
| Model Comparison | ❌ | ✅ |

---

## 🖥️ Technologies Used
- Python 3.x  
- PyTorch & Torchvision  
- Gradio (Interactive interface)  
- NumPy  
- PIL (Image processing)  
- Matplotlib (Heatmaps & visualization)  

---

## 📁 Project Structure

NeuroVision-AI/
│
├── version1/
│ ├── notebooks/ # Training notebook for Version 1
│ ├── models/ # Trained model for tumor detection
│ ├── data/ # Sample dataset (CT images)
│ ├── results/ # Outputs / predictions
│ └── README.md # Version 1 documentation
│
├── version2/
│ ├── notebooks/ # 4 training notebooks (tumor & stroke, CT & MRI)
│ ├── models/ # 4 trained models
│ ├── data/ # Expanded dataset (CT + MRI)
│ ├── results/ # Outputs / predictions
│ ├── Src/ # Gradio app & source code
│ └── README.md # Version 2 documentation
│
├── README.md # Main README (this file)
├── requirements.txt
└── .gitignore

---

## 🎥 Demo Videos
- **Version 1:** [Add video link here]  
- **Version 2:** [Add video link here]  

---

## ⚠️ Disclaimer
This project is for **educational and research purposes only**.  
It is **not intended for clinical use or medical diagnosis**.  

---

## 🏆 Conclusion
**NeuroVision AI** demonstrates a clear evolution in medical AI systems:  
- Strong implementation of deep learning in healthcare  
- Practical, interactive Gradio interface  
- Multi-model system capable of cross-modality analysis  
- Scalable, well-structured project ready for research and experimentation  

---

## 👨‍💻 Author
Developed as part of an **AI & Medical Imaging Research Project**.  
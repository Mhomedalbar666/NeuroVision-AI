# 🧠 NeuroVision AI – Version 2  
### Advanced Multi-Model System for Brain Tumor & Stroke Detection

---

## 📌 Overview
**NeuroVision AI (Version 2)** is an advanced medical imaging system that significantly improves upon Version 1 by introducing a **multi-model architecture** for detecting brain tumors and stroke using both **CT and MRI images**.

The system is designed to analyze medical images, compare multiple models simultaneously, and provide accurate predictions along with visual explanations and medical guidance.

---

## 🚀 Key Improvements Over Version 1
- 🧠 Detection of both **Brain Tumor and Stroke**  
- 🤖 Multiple specialized models (4 models)  
- 🧪 Support for **CT and MRI imaging modalities**  
- 📊 Comparative analysis between models  
- 🔥 Enhanced heatmap visualization  
- 🌐 Multi-language support (English / Arabic)  
- 🩺 Advanced medical recommendations  

---

## 🧠 Multi-Model Architecture

The system includes four specialized models:

- **CT – Tumor Detection**  
  `model_tumor_weights.pth`

- **CT – Stroke Detection**  
  `modelv2_weights_another.pth`

- **MRI – Stroke Detection**  
  `modelv1_mr_weights.pth`

- **MRI – Tumor Detection**  
  `model_weights_tumor_mri_second.pth`

---

## ⚙️ System Workflow

1. User selects image type:
   - CT or MRI  

2. Based on the selection:
   - The system loads the appropriate models  

3. The uploaded image is processed by:
   - Two models (same modality, different tasks)  

4. Outputs include:
   - Classification results  
   - Probability scores  
   - Heatmap visualization  

---

## 🔬 Model Comparison Strategy
To study the impact of imaging modality:

- Each image is analyzed by **multiple models simultaneously**  
- Comparison between:
  - Tumor vs Stroke detection  
  - CT vs MRI performance  

This allows deeper understanding of:
- Model behavior  
- Imaging differences  
- Diagnostic accuracy  

---

## 🔥 Heatmap Enhancement
- Improved Grad-CAM visualization  
- Color intensity increases toward abnormal regions  
- Red areas indicate higher model attention  

---

## 📈 Performance Improvements
- Increased dataset size  
- Higher number of training epochs  
- Improved preprocessing and transformations  
- Optimized model architectures  

---

## 🖥️ User Interface (Gradio)

The system provides a powerful interactive interface:

- Upload CT or MRI image  
- Select language (Arabic / English)  
- Enter patient age  
- View:
  - Predictions  
  - Probabilities  
  - Heatmaps  

---

## 🌐 Multi-Language Support
- English  
- Arabic  

Users can switch language dynamically from the interface.

---

## 🩺 Smart Medical Recommendations
- Recommendations based on:
  - Age  
  - Prediction results  
- Advice intensity increases if:
  - Both tumor and stroke are detected  

---

## 📍 External Medical Resources
The system provides trusted links to:
- Medical consultation platforms  
- Specialized healthcare centers  

---

## 📁 Project Structure

version2/
│
├── notebooks/
│ ├── part1.ipynb
│ └── part2.ipynb
│
├── models/
│ ├── ct_tumor.pth
│ ├── ct_stroke.pth
│ ├── mri_stroke.pth
│ └── mri_tumor.pth
│
├── data/
│ └── sample_images/
│
├── results/
│ ├── ct_results/
│ └── mri_results/
│
├── src/
│ └── app.py
│
└── README.md



---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/app.py

⚠️ Disclaimer

This system is for:

Educational purposes
Research use only

It is not a replacement for professional medical diagnosis.

🏆 Conclusion

Version 2 represents a significant advancement in:

Model performance
System design
Medical insight

It demonstrates a complete and scalable AI-powered medical imaging solution.
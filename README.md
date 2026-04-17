# 🌾 Rice Leaf Disease Detection using Deep Learning (CNN vs Swin Transformer with XAI)

## 📌 Project Overview
This project focuses on automatic detection of rice leaf diseases using Deep Learning techniques. It compares two models:
- **ResNet50 (CNN)**
- **Swin Transformer (Vision Transformer)**

To ensure transparency and trust in predictions, Explainable AI (XAI) techniques such as **Grad-CAM++** and **Gradient SHAP** are applied.

The system is designed to assist farmers and agricultural experts in early and accurate disease identification, improving crop yield and reducing losses.

---

## 🎯 Objectives
- Compare performance of CNN (ResNet50) and Swin Transformer
- Improve rice disease detection accuracy using Deep Learning
- Use class-balanced dataset for fair evaluation
- Apply Explainable AI for model interpretability
- Ensure biological correctness of predictions

---

## 📊 Dataset Used
**Mendeley Rice Disease & Pest Dataset**

🔗 Link: https://data.mendeley.com/datasets/vwv3nry3wr/1

### Details:
- Original Images: 2,769
- Augmented Images: 19,128
- Classes:
  - Rice Blast
  - Leaf Blast
  - Leaf Stripes
  - Leaffolder
  - Tungro
  - Insect damage
  - Healthy leaves
  - General rice category

### Augmentation Techniques:
- Rotation
- Flipping
- Cropping
- Scaling
- Color adjustment

---

## 🧠 Models Used

### 1. ResNet50 (CNN)
- Deep Residual Network
- Uses skip connections
- Partial fine-tuning
- Good for local feature extraction

### 2. Swin Transformer
- Shifted Window Self-Attention
- Captures global + local features
- Better for complex disease patterns
- Faster convergence

---

## 🔍 Explainable AI (XAI)
To improve transparency:

### ✔ Grad-CAM++
- Highlights disease-affected regions in image
- Shows where model is focusing

### ✔ Gradient SHAP
- Explains feature importance
- Identifies texture and color influence

---

## ⚙️ Methodology
1. Dataset collection and preprocessing
2. Class balancing of dataset
3. Image resizing (224×224)
4. Normalization (ImageNet standards)
5. Training ResNet50 and Swin Transformer
6. Model evaluation using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
7. Applying XAI for interpretation

---

## 📈 Results

| Model           | Accuracy | F1 Score | Epochs |
|----------------|----------|----------|--------|
| ResNet50       | 93.72%   | 0.9425   | 30     |
| Swin Transformer | 95.33%   | 0.9594   | 20     |

### Key Observations:
- Swin Transformer outperforms ResNet50
- Faster convergence in Transformer model
- Better performance on complex disease patterns

---

## 🧾 Key Findings
- Transformers capture long-range spatial patterns better than CNNs
- Class-balanced dataset improves fairness
- XAI confirms model focuses on actual disease regions
- Reduces false predictions and improves trust

---

## 🚀 Future Scope
- Deployment on mobile and IoT devices
- Real-time field disease detection
- Disease severity estimation
- Use of diffusion models for data augmentation
- Domain adaptation for real-world environments

---

## 💻 Tech Stack
- Python
- TensorFlow / PyTorch
- OpenCV
- NumPy, Pandas
- Matplotlib
- Grad-CAM++, SHAP

---

## 📌 Project Impact
- Helps farmers in early disease detection
- Reduces crop loss
- Improves agricultural productivity
- Enables smart farming using AI

---

## 👩‍💻 Author
**Pranali Patil**  
**Piyush Patil**  
**Sayali Pawar**  

PCCOE, Pune  

---

## 📜 License
This project is for academic purposes only.

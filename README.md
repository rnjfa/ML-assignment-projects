# 📚 Machine Learning Project Collection

This repository contains a collection of four Machine Learning projects developed by **Fateme Ranjbaran** for an advanced Machine Learning course. The projects cover a wide range of ML topics, including fraud detection, time series forecasting, image classification using CNNs, and transfer learning using pre-trained models.

---

## 🚀 **Projects Overview**

### 1️⃣ [Fraud Detection with MLP](./Fraud_Detection)
- **Objective:** Classify financial transactions as fraudulent or legitimate using an MLP classifier.
- **Dataset:** [Credit Card Transaction Dataset] (https://www.kaggle.com/code/sharmabhilash/credit-card-fraud-detection).
- **Techniques Used:**
  - Data Cleaning and Handling Missing Values.
  - Feature Selection using Random Forest Importance (Top 58 features selected).
  - Handling Class Imbalance using Class Weights and Threshold Optimization.
  - Model Evaluation: Confusion Matrix, Precision, Recall, F1-Score.
- **Model Architecture:**
  - MLP with 2 hidden layers (50 and 30 neurons), ReLU activations, and sigmoid output.
- **Key Results:**
  - Best Threshold: **0.83**, Best Precision: **96.1%**.
  - Final Evaluation:
    - F1-Score: 0.493  
    - Recall: 0.506  
    - Precision: 0.712  

---

### 2️⃣ [Weather Prediction with LSTM](./Weather_Prediction_LSTM)
- **Objective:** Predict rainfall for the next day using historical weather data.
- **Dataset:** [WeatherAUS Dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package).
- **Techniques Used:**
  - Time Series Processing with Sliding Windows (5 Days Lookback).
  - LSTM Model Design using TensorFlow/Keras.
  - Comparison with GRU Models for Performance Analysis.
  - Evaluation Metrics: Confusion Matrix, Precision, Recall, F1-Score.
- **Key Results:**
  - LSTM Accuracy: **~90%**
  - GRU Accuracy: **~99%** (with reduced overfitting).
  - Explored different time windows (3, 5, 10 days), achieving highest accuracy with 5-day sequences.

---

### 3️⃣ [Fashion MNIST Image Classification with CNN](./Fashion_MNIST_Classification)
- **Objective:** Classify fashion items using Convolutional Neural Networks.
- **Dataset:** Fashion MNIST dataset (60,000 training and 10,000 testing images).
- **Techniques Used:**
  - Data Normalization and Visualization.
  - CNN Model Architecture using TensorFlow/Keras.
  - Evaluation Metrics: Confusion Matrix, Precision, Recall, F1-Score.
- **Key Results:**
  - Validation Accuracy: **91.74%**
  - Final Evaluation:
    - F1-Score: 0.918  
    - Recall: 0.917  
    - Precision: 0.919  

---

### 4️⃣ [Image Classification with AlexNet (Transfer Learning)](./AlexNet_TransferLearning)
- **Objective:** Perform image classification using the AlexNet model with Transfer Learning.
- **Dataset:** Custom Dataset with 3 Classes: `bear`, `gorilla`, `other`.
- **Techniques Used:**
  - Transfer Learning using Pre-trained AlexNet on ImageNet.
  - Fine-tuning final layers and freezing pre-trained layers.
  - Model Evaluation using Accuracy and Confusion Matrix.
- **Key Results:**
  - Achieved **95%+ accuracy** on test images.
  - Efficient classification using transfer learning with minimal training time.

---

## 📚 **Technologies & Libraries Used**
- Python 3.x
- NumPy, Pandas
- Scikit-Learn
- TensorFlow / Keras
- Matplotlib, Seaborn
- Scikit-Image
- OpenCV

---


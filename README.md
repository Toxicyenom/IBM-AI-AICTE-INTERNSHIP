# 🗑️ Smart Waste Segregation using EfficientNet + Custom CNN (Garbage Classification)

A deep learning project that classifies waste images into 6 recyclable categories using a hybrid CNN model built by fine-tuning EfficientNetV2B2 with custom Convolutional Neural Network layers. The project includes a web-based interface using Gradio with support for image upload and real-time webcam capture.

---

## Features

-  **Hybrid Architecture** – Combines EfficientNetV2B2 with fine-tuned custom CNN layers for superior accuracy and robustness.
-  **Transfer Learning Enabled** – Uses pre-trained EfficientNet weights for faster convergence and better generalization.
-  **High-Precision Garbage Classification** – Classifies waste into six categories: Cardboard, Glass, Metal, Paper, Plastic, and Trash.
-  **Webcam + Upload Prediction** – Real-time prediction using either webcam or uploaded image directly in browser (via Gradio).
-  **Live Inference via Gradio UI** – Seamless user interface for testing any image instantly, no code required.
-  **Performance Monitoring** – Includes loss/accuracy graphs, learning curves, and training logs.
-  **Confusion Matrix** – Visual breakdown of model predictions vs true labels.
-  **Detailed Classification Report** – Outputs precision, recall, F1-score for each class.
-  **Model Persistence** – Easily save and reload the best performing `.keras` models.
-  **Organized Dataset Pipeline** – Structured training/validation/test data loading with augmentation.
-  **Code Modularity** – Separated files for training, deployment, evaluation, and webcam utility.

---
## ⚙️ Model Architecture 

<p align="center">
  <img src="https://github.com/Salaar-Saaiem/Garbage-Classification-using-ML/blob/main/Assets/Architecture%20Diagram.png?raw=true" alt="Model Architecture" width="700"/>
</p>
<p>
The visual above represents the complete training and evaluation pipeline. It uses EfficientNetV2B2 as a feature extractor, followed by custom CNN layers. The model is initially trained with a frozen base, then fine-tuned, and finally evaluated on multiple metrics like accuracy, loss curves, and confusion matrix.
</p>

## 🧩 Model Flow Diagram

<p align="center">
  <img src="https://github.com/Salaar-Saaiem/Garbage-Classification-using-ML/blob/main/Assets/Flow%20Diagram.png?raw=true" alt="Model Flow Diagram" width="700"/>
</p>

---

## 📂 Target Classes

- 📦 Cardboard  
- 🧪 Glass  
- ⚙️ Metal  
- 📄 Paper  
- 🧴 Plastic  
- 🚮 Trash  

---

## Tech Stack

-  **TensorFlow / Keras** – Core deep learning framework used for building, training, fine-tuning, and saving models.
-  **EfficientNetV2B2** – Transfer learning backbone pre-trained on ImageNet, integrated for better performance and faster convergence.
-  **Custom CNN Layers** – Tailored layers added over EfficientNet for domain-specific fine-tuning and improved accuracy.
-  **tf.keras.preprocessing & ImageDataGenerator** – For real-time image augmentation, scaling, and train-validation-test pipeline.
-  **Matplotlib & Seaborn** – For visualizing performance metrics like learning curves, confusion matrix, and prediction results.
-  **Scikit-learn** – For generating classification reports, precision, recall, and F1-score.
-  **Gradio** – For browser-based UI that supports both image upload and live webcam input for real-time predictions.
-  **Python 3.10** – Programming language used throughout the entire project.
-  **NumPy & Pandas** – For efficient numerical operations and dataset handling.
-  **Jupyter Notebook** – For model experimentation, prototyping, and performance evaluation.



---

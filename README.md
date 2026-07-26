# Facial Emotion Recognition using CNN

A deep learning project that recognizes facial emotions in real time using a Convolutional Neural Network (CNN). The model was developed using TensorFlow and Keras, trained on a publicly available Kaggle dataset, and integrated with OpenCV for webcam-based emotion detection.

---

## Features

- Real-time facial emotion detection using a webcam
- Face detection using OpenCV Haar Cascades
- Seven emotion classification
- Convolutional Neural Network (CNN) built with TensorFlow and Keras
- Image preprocessing and data augmentation
- Early Stopping and Model Checkpointing during training
- Live prediction display on detected faces

---

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Google Colab

---

## Model Architecture

The CNN consists of:

- 4 Convolutional Layers
- Batch Normalization
- Max Pooling
- Fully Connected Dense Layers
- Dropout
- Softmax Output Layer

---

## Dataset

The model was trained using a publicly available facial emotion dataset from Kaggle.

Input Image:

- 48 × 48 pixels
- Grayscale

Emotion Classes:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

## Training Techniques

- Data Augmentation
- Adam Optimizer
- Categorical Crossentropy Loss
- Early Stopping
- Model Checkpoint
- Batch Normalization
- Dropout

---

## Model Performance

- Validation Accuracy: **~60%**
- Validation Loss: **~1.10**

The model provides reliable predictions for a student project and demonstrates the complete workflow of training, evaluating, saving, and deploying a CNN for real-time facial emotion recognition.

---

## Project Structure

```
train.py                 # Model training
detect_emotion.py        # Real-time webcam prediction
best_emotion_model.h5    # Best saved model
README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Keven-0129/facial-emotion-recognition.git
```

Install the required libraries

```bash
pip install tensorflow opencv-python numpy matplotlib
```

---

## Running the Project

### Train the model

```bash
python train.py
```

### Run real-time emotion detection

```bash
python detect_emotion.py
```

Press **Q** to exit the webcam window.

---

## Results

### Training Accuracy

<img width="352" height="320" alt="accuracy" src="https://github.com/user-attachments/assets/55038ca3-6cc1-4bf0-a96b-637fc3ab698f" />


### Training Loss

<img width="329" height="320" alt="loss" src="https://github.com/user-attachments/assets/2b2a5da8-a0a6-4037-babb-45d1fa9e77cc" />


### Real-Time Detection

<img width="1600" height="900" alt="webcam_detection" src="https://github.com/user-attachments/assets/fda3a9d5-144f-4139-bbf5-98aa0d251071" />


---

## Future Improvements

- Improve model accuracy using transfer learning
- Train on a larger facial expression dataset
- Deploy as a web application
- Improve face detection using deep learning-based detectors
- Optimize the model for faster real-time inference

---

## Author

Nikush Sharma

# 💤 Driver Drowsiness Detection System (CNN + OpenCV)

This project implements a **real-time driver drowsiness detection system** using a lightweight Convolutional Neural Network (VGG-based), classical computer vision methods (Haar cascades), and PyTorch for classification.

The system captures live video from a webcam, detects the driver’s face and eye regions, and classifies the eye state (open/closed). If the eyes remain closed for too long, it triggers an audible alarm.

---

## 🚀 Features

- Real-time face and eye detection using OpenCV Haar cascades  
- Eye state classification (open/closed) using a trained VGG model  
- Alarm system (using `pygame`) to alert when eyes are closed too long  
- Robust detection loop with scoring mechanism to avoid false positives  
- Easy-to-extend modular design (`VGG.py`, `TrainingModules.py`, `DataPreprocessing.py`)  

---

## 📁 Directory Structure

```
project-root/
├── main.py                  # Main detection loop (run this file)
├── VGG.py                   # CNN model definition
├── TrainingModules.py       # Prediction function
├── DataPreprocessing.py     # Image transforms (e.g., test_transform)
├── checkpoint/              # Folder for saved model weights
│   └── vgg.16.pth           # Trained model
├── alarm.wav                # Alarm sound played on drowsiness
├── README.md                # This file
```

---

## 🛠️ Requirements

- Python 3.8 or 3.9 recommended  
- PyTorch  
- OpenCV  
- torchvision  
- pygame  
- matplotlib  
- Pillow (PIL)  
- tqdm  
- torchprofile  
- scikit-learn  

### ✅ Install via pip

```bash
pip install torch torchvision opencv-python pygame matplotlib tqdm pillow scikit-learn torchprofile
```

---

## 🧪 Running the Project

```bash
python main.py
```

> Press `Q` to quit the webcam window.

---

## 📷 How it Works

1. Webcam captures frames in real time.  
2. Haar cascades detect the face and eyes.  
3. The cropped eye region is preprocessed and passed to a CNN model.  
4. The model classifies the eye state:  
   - **Closed**: Increases `score`  
   - **Open**: Decreases `score`  
5. If `score > 20`, the alarm is triggered via `pygame`.

---

## 🧠 Model Details

- The CNN is a lightweight **VGG-based** model trained on eye state data.  
- It is loaded from `checkpoint/vgg.16.pth`.  
- You can retrain or replace the model by modifying `VGG.py`.

---

## 🔊 Alarm System

- The `alarm.wav` is played when the driver’s eyes remain closed for a defined threshold of frames (`score > 20`).  
- Sound alert uses `pygame.mixer` to notify the driver.

---

## 📌 Notes

- Ensure the driver’s eyes are clearly visible to the camera.  
- Works best under consistent lighting conditions.  
- Thresholds (`score > 20`) can be tuned for stricter or looser alert criteria.

---

## 🧾 License

This project is intended for **academic and research purposes** only.

---

## 🙋‍♂️ Acknowledgements

- [OpenCV](https://opencv.org/) for face and eye detection  
- [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/stable/index.html)  
- Haar cascade classifiers provided by OpenCV  

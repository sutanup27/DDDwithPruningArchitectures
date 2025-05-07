# 💤 Driver Drowsiness Detection System (CNN + OpenCV + Architecture Pruning)

This project implements a real-time driver drowsiness detection system using a lightweight Convolutional Neural Network (VGG-based), Dlib's facial landmark detection, and PyTorch for classification. Utilised pruning techniques to reduce the model's size and performance in real-time.

The system captures live video from a webcam, detects the driver’s face, extracts eye regions using Dlib’s 68-point facial landmark predictor, and classifies the eye state (open/closed). If the eyes remain closed for too long, an audible alarm triggers.



---

## 🚀 Features

- Real-time face and eye detection using OpenCV Dlib  
- Eye state classification (open/closed) using a trained VGG model  
- Alarm system (using `pygame`) to alert when eyes are closed for too long  
- Robust detection loop with scoring mechanism to avoid false positives  
- Easy-to-extend modular design (`VGG.py`, `TrainingModules.py`, `DataPreprocessing.py`)  

---

## 📁 Directory Structure

```
project-root/
├── main.py                          # Main detection loop (run this file)
├── VGG.py                           # CNN model definition (VGG-based architecture)
├── TrainingModules.py               # Training, validation, and prediction functions
├── DataPreprocessing.py             # Image transforms and dataset preparation
├── DataPreview.py                   # Visualization of dataset images and metadata
├── DlibTest.py                      # Test script using Dlib for facial landmarks
├── Model_Evaluation.py              # Evaluation metrics and reporting
├── PruneEvaluator.py                # Evaluate pruned models
├── PruneTest.py                     # Testing pruned CNNs
├── PruneTrain.py                    # Training loop for pruned CNNs
├── PruneViewer.py                   # Visualizer for pruned architectures and stats
├── Test.py                          # Custom test cases for quick validation
├── Train.py                         # General training loop
├── Util.py                         # Helper functions (utilities)
├── Viewer.py                        # Output and performance visualization tools
├── DDD.ipynb                        # Jupyter notebook for interactive experiments
├── alarm.wav                        # Alarm sound played on drowsiness detection
├── requirement.txt                  # Required Python libraries
├── shape_predictor_68_face_landmarks.dat  # Dlib model for facial landmarks
├── README.md                        # Project documentation and instructions
├── .gitignore                       # Git ignore rules
├── checkpoint/                      # Folder for saved model checkpoints and results
│   ├── accuracies.pkl               # Pickled accuracy scores
│   ├── sparsities.pkl               # Pickled sparsity metrics
│   ├── sensitivity_scan.png         # Plot showing sensitivity scan
│   ├── vgg_metrics.txt              # Metrics summary for VGG models
│   ├── vgg_mrl_99.0929946899414.pth         # Saved VGG model with 99.09% accuracy
│   ├── vgg_mrl_CP_98.90.txt                  # CP model metrics
│   ├── vgg_mrl_CP_98.90452575683594.pth      # Saved CP model weights
│   ├── vgg_mrl_fgp_99.039.txt                # FGP model metrics
│   ├── vgg_mrl_fgp_99.03999328613281.pth     # Saved FGP model weights
│   └── output/                      # Subdirectory for generated outputs
│       └── (your generated plots, logs, etc.)

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
pip install -r requirement.txt
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

import cv2
import numpy as np
import random
import torch
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *
from pygame import mixer
from VGG import VGG
from TrainingModules import predict
from DataPreprocessing import test_transform
from PIL import Image
import dlib

# Fix randomness
seed = 0
random.seed(seed)
np.random.seed(seed)

mixer.init()
sound = mixer.Sound('alarm.wav')

# Dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

model = VGG()
model = torch.load("checkpoint/vgg_mrl_99.0929946899414.pth", map_location=torch.device('cpu'),weights_only=False)  # Use 'cpu' if necessary
model.eval()
lbl = ['Close', 'Open']

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for face in faces:
        x, y, x1, y1 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 3)
        landmarks = predictor(gray, face)

        # Left eye (points 36-41), Right eye (points 42-47)
        left_eye_pts = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye_pts = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

        # Get bounding boxes around eyes
        for eye_pts in [left_eye_pts, right_eye_pts]:
            ex = min([p[0] for p in eye_pts])
            ey = min([p[1] for p in eye_pts])
            ew = max([p[0] for p in eye_pts]) - ex
            eh = max([p[1] for p in eye_pts]) - ey

            eye = frame[ey:ey + eh, ex:ex + ew]
            if eye.size == 0:
                continue
            eye_pil = Image.fromarray(cv2.cvtColor(eye, cv2.COLOR_BGR2RGB))
            eye_transformed = test_transform(eye_pil)
            prediction = predict(model, eye_transformed)

            if prediction == 0:
                cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                score += 1
                if score > 20:
                    try:
                        sound.play(maxtime=5000)
                    except:
                        pass
            else:
                score -= 1
                if score < 0:
                    score = 0
                cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

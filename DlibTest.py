import cv2
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *


import cv2
import dlib

# Load image
image_path = '../mrleyedataset/image_lc.jpg'  # Change this to your image path
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Detect faces
faces = detector(gray)

# Loop through faces
for face in faces:
    landmarks = predictor(gray, face)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    # Optionally draw the face bounding box
    cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

# Show and save result
cv2.imshow("Landmarks", image)
cv2.imwrite("chekpoint/output/output_landmarks.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

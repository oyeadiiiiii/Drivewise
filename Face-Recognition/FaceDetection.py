import cv2
import numpy as np
from os import path

name = input("Type your Name ")

cap = cv2.VideoCapture(1)  # Changed camera index to 0
classifier = cv2.CascadeClassifier("C:\\DriveWise-main\\Face-Recognition\\haarcascade_frontalface_default.xml")

count = 50
face_list = []
face_img = None  # Initialize face_img outside the loop

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = classifier.detectMultiScale(gray)
    
    # Only proceed if a clear picture is detected
    if len(faces) == 1:
        x, y, w, h = faces[0]
        if w > 100 and h > 100:  # Adjust this threshold as needed
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (100, 100))
            face_list.append(face_img.flatten())
            count -= 1
            print("Loaded images:", 50 - count)
            if count <= 0:
                break

    cv2.imshow("video", face_img if face_img is not None else frame)  # Display frame if no face detected
    if cv2.waitKey(1) > 30:
        break

face_list = np.array(face_list)
name_list = np.full((len(face_list), 1), name)

total = np.hstack([name_list, face_list])
if path.exists("faces.npy"):
    data = np.load("faces.npy")
    data = np.vstack([data, total])
else:
    data = total
np.save("faces.npy", data)

cap.release()
cv2.destroyAllWindows()

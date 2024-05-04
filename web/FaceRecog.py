import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def facerecog():
    # Load the data
    data = np.load(r"C:\DriveWise-main\faces.npy")

    cap = cv2.VideoCapture(1)
    classifier = cv2.CascadeClassifier(r"C:\DriveWise-main\Face-Recognition\haarcascade_frontalface_default.xml")

    X = data[:, 1:].astype(int)
    y = data[:, 0]

    model = KNeighborsClassifier(n_neighbors=4)
    model.fit(X, y)

    predicted_name = None

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame from camera/stream end")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = classifier.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (100, 100))

                flat = face_img.flatten()
                res = model.predict([flat])
                label = str(res[0])
                
                predicted_name = label
                
                break

            if len(faces) > 0:
                break

            if cv2.waitKey(1) > 30:
                break

        except Exception as e:
            print(f"An error occurred: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()

    return predicted_name

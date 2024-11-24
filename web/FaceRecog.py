import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def facerecog():
    try:
        # Load the data
        data = np.load(r"faces.npy")

        cap = cv2.VideoCapture(1)
        classifier = cv2.CascadeClassifier(r"web\haarcascade_frontalface_default.xml")

        X = data[:, 1:].astype(int)  # Extract features (pixel values)
        y = data[:, 0]  # Extract labels (person IDs or names)

        model = KNeighborsClassifier(n_neighbors=4, metric='euclidean')
        model.fit(X, y)

        # Set a distance threshold for recognizing a face
        threshold = 10000  # Adjust this value based on your training data and testing

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame from camera/stream end")
                break

            frame = cv2.flip(frame, 1)  # Flip the frame for a mirror effect
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

            faces = classifier.detectMultiScale(gray)

            if len(faces) == 0:
                return "No Face"

            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (100, 100))  # Resize to match the training size

                flat = face_img.flatten()  # Flatten the face image to feed into the model

                # Get the nearest neighbors and their distances
                distances, indices = model.kneighbors([flat], n_neighbors=1)
                nearest_distance = distances[0][0]

                # Determine if the face is "unknown"
                if nearest_distance > threshold:
                    return "Unknown Driver"
                else:
                    return str(model.predict([flat])[0])  # Recognized name

    except Exception as e:
        print(f"An error occurred in facerecog: {e}")
        return None

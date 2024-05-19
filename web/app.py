import threading
import time
from flask import Flask, Response, render_template, jsonify, request
import cv2
import numpy as np
from os import path
from sklearn.neighbors import KNeighborsClassifier
from act import main

app = Flask(__name__)

# Load facial data if available
data_file = "faces.npy"
if path.exists(data_file):
    data = np.load(data_file)
    X = data[:, 1:].astype(int)
    y = data[:, 0]
else:
    X, y = np.array([]), np.array([])

# Initialize KNN model
model = KNeighborsClassifier(n_neighbors=4)
if len(X) > 0:
    model.fit(X, y)

# Global variables
prev_name = ""
update_interval = 1  # in seconds

def facerecog(frame, model):
    if model is None:
        return None

    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_img = cv2.resize(gray[y:y+h, x:x+w], (100, 100)).flatten()
        predicted_name = model.predict([face_img])[0]
        return str(predicted_name)
    else:
        return None

def update_driver_name():
    global prev_name
    while True:
        try:
            # Load facial data
            data = np.load(data_file)
            X = data[:, 1:].astype(int)
            y = data[:, 0]
            
            # Check if data is available for training
            if len(X) > 0:
                # Initialize KNN model
                model = KNeighborsClassifier(n_neighbors=4)
                # Fit the model with data
                model.fit(X, y)

                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                if ret:
                    name = facerecog(frame, model)
                    cap.release()
                    
                    # Update name only if it differs from the previous one
                    if name != prev_name:
                        prev_name = name
            time.sleep(update_interval)  # Sleep for update_interval seconds
        except Exception as e:
            print(f"Error updating driver's name: {e}")

# Start the thread for updating the driver's name
update_thread = threading.Thread(target=update_driver_name)
update_thread.daemon = True
update_thread.start()

def gen_frames_act():
    for frame, _ in main():
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed_act')
def video_feed_act():
    try:
        return Response(gen_frames_act(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video_feed_act: {e}")
        return str(e), 500

@app.route('/get_driver_name')
def get_driver_name():
    global prev_name
    try:
        return jsonify(driverName=prev_name)
    except Exception as e:
        print(f"Error in get_driver_name: {e}")
        return jsonify(driverName=None)

@app.route('/register_driver', methods=['POST'])
def register_driver():
    try:
        data = request.get_json()
        name = data.get('name')
        
        # Your FaceDetection code here...
        cap = cv2.VideoCapture(0)
        classifier = cv2.CascadeClassifier(r'C:\DriveWise\haarcascade_frontalface_default.xml')
        
        count = 50
        face_list = []
        face_img = None

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = classifier.detectMultiScale(gray)
            
            if len(faces) == 1:
                x, y, w, h = faces[0]
                if w > 100 and h > 100:
                    face_img = gray[y:y + h, x:x + w]
                    face_img = cv2.resize(face_img, (100, 100))
                    face_list.append(face_img.flatten())
                    count -= 1
                    print("Loaded images:", 50 - count)
                    if count <= 0:
                        break

            cv2.imshow("video", face_img if face_img is not None else frame)
            if cv2.waitKey(1) > 30:
                break

        face_list = np.array(face_list)
        name_list = np.full((len(face_list), 1), name)

        total = np.hstack([name_list, face_list])
        if path.exists("faces.npy"):
            existing_data = np.load("faces.npy")
            data = np.vstack([existing_data, total])
        else:
            data = total

        np.save("faces.npy", data)

        cap.release()
        cv2.destroyAllWindows()

        return jsonify(success=True)
    
    except Exception as e:
        print(f"Error registering driver: {e}")
        return jsonify(success=False, error=str(e)), 500

@app.route('/state_feed')
def state_feed():
    def generate():
        try:
            for _, state in main():
                yield f"data: {state}\n\n"
        except Exception as e:
            print(f"Error in state_feed: {e}")
            yield f"data: Error: {e}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)

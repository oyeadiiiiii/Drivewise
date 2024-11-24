import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, request
from os import path
import threading
from act import main  # Assuming this handles video feed and other actions
from FaceRecog import facerecog  # Assuming this is your face recognition function


app = Flask(__name__)

prev_name = ""
current_name = None  # Global variable to store current recognized driver name

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
    global current_name
    try:
        # Return the current recognized name or its state
        return jsonify(driverName=current_name)
    except Exception as e:
        print(f"Error in get_driver_name: {e}")
        return jsonify(error=str(e)), 500




@app.route('/register_driver', methods=['POST'])
def register_driver():
    try:
        data = request.get_json()
        name = data.get('name')
        
        # Your FaceDetection code here...
        cap = cv2.VideoCapture(1)
        classifier = cv2.CascadeClassifier(r"web\haarcascade_frontalface_default.xml")
        
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

def recognize_driver():
    global current_name
    while True:
        try:
            name = facerecog()  # Get the recognized name from facerecog()
            if name != current_name:
                current_name = name  # Update the global variable only when it changes
                print(f"Updated current_name to: {current_name}")  # Debugging line
        except Exception as e:
            print(f"Error in recognize_driver: {e}")




if __name__ == '__main__':
    # Run facerecog and main functions in separate threads
    t1 = threading.Thread(target=main)  # Main video feed handling
    t2 = threading.Thread(target=recognize_driver)  # Face recognition handling
    
    t1.start()
    t2.start()

    app.run(debug=True)

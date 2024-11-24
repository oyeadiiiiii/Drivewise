import cv2

def find_default_camera():
    # Try to open the first 10 cameras
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Default camera index: {i}")
            cap.release()
            return i
    print("Unable to find the default camera.")
    return None

# Call the function to find the default camera index
default_camera_index = find_default_camera()

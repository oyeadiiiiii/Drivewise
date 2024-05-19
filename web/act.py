import cv2
import numpy as np
import mediapipe as mp
import time
import argparse

from Utils import get_face_area
from Eye_Dector_Module import EyeDetector
from Pose_Estimation_Module import HeadPoseEstimator
from Attention_Scorer_Module import AttentionScorer

def main():
    parser = argparse.ArgumentParser(description='Driver State Detection')

    parser.add_argument('-c', '--camera', type=int, default=0, metavar='', help='Camera number, default is 0 (webcam)')
    parser.add_argument('--show_fps', action='store_true', help='Show the actual FPS of the capture stream, default is true')
    parser.add_argument('--show_proc_time', action='store_true', help='Show the processing time for a single frame, default is true')
    parser.add_argument('--show_eye_proc', action='store_true', help='Show the eyes processing, default is false')
    parser.add_argument('--show_axis', action='store_true', help='Show the head pose axis, default is true')
    parser.add_argument('--verbose', action='store_true', help='Prints additional info, default is false')

    parser.add_argument('--ear_thresh', type=float, default=0.15, metavar='', help='Sets the EAR threshold for the Attention Scorer, default is 0.15')
    parser.add_argument('--ear_time_thresh', type=float, default=2, metavar='', help='Sets the EAR time (seconds) threshold for the Attention Scorer, default is 2 seconds')
    parser.add_argument('--gaze_thresh', type=float, default=0.015, metavar='', help='Sets the Gaze Score threshold for the Attention Scorer, default is 0.2')
    parser.add_argument('--gaze_time_thresh', type=float, default=2, metavar='', help='Sets the Gaze Score time (seconds) threshold for the Attention Scorer, default is 2 seconds')
    parser.add_argument('--pitch_thresh', type=float, default=20, metavar='', help='Sets the PITCH threshold (degrees) for the Attention Scorer, default is 30 degrees')
    parser.add_argument('--yaw_thresh', type=float, default=20, metavar='', help='Sets the YAW threshold (degrees) for the Attention Scorer, default is 20 degrees')
    parser.add_argument('--roll_thresh', type=float, default=20, metavar='', help='Sets the ROLL threshold (degrees) for the Attention Scorer, default is 30 degrees')
    parser.add_argument('--pose_time_thresh', type=float, default=2.5, metavar='', help='Sets the Pose time threshold (seconds) for the Attention Scorer, default is 2.5 seconds')

    args = parser.parse_args()

    if args.verbose:
        print(f"Arguments and Parameters used:\n{args}\n")

    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)
        except:
            print("OpenCV optimization could not be set to True, the script may be slower than expected")

    detector = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)
    eye_detector = EyeDetector(show_processing=args.show_eye_proc)
    head_pose_estimator = HeadPoseEstimator(show_axis=args.show_axis)
    t0 = time.perf_counter()
    scorer = AttentionScorer(t_now=t0, ear_thresh=args.ear_thresh, gaze_time_thresh=args.gaze_time_thresh,
                             roll_thresh=args.roll_thresh, pitch_thresh=args.pitch_thresh,
                             yaw_thresh=args.yaw_thresh, ear_time_thresh=args.ear_time_thresh,
                             gaze_thresh=args.gaze_thresh, pose_time_thresh=args.pose_time_thresh,
                             verbose=args.verbose)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    i = 0
    time.sleep(0.01)
    eyes_closed_time = 0
    prev_state = ""  # Initialize previous state
    proper_driving_flag = False  # Initialize proper driving flag
    distracted_flag = False
    asleep_flag = False

    while True:
        t_now = time.perf_counter()
        fps = i / (t_now - t0)
        if fps == 0:
            fps = 10

        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame from camera/stream end")
            break

        if args.camera == 0:
            frame = cv2.flip(frame, 2)

        e1 = cv2.getTickCount()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_size = frame.shape[1], frame.shape[0]
        gray = np.expand_dims(cv2.bilateralFilter(gray, 5, 10, 10), axis=2)
        gray = np.concatenate([gray, gray, gray], axis=2)

        lms = detector.process(gray).multi_face_landmarks

        description = ""

        if lms:
            landmarks = _get_landmarks(lms)
            ear = eye_detector.get_EAR(frame=gray, landmarks=landmarks)
            tired, perclos_score = scorer.get_PERCLOS(t_now, fps, ear)
            gaze = eye_detector.get_Gaze_Score(frame=gray, landmarks=landmarks, frame_size=frame_size)
            frame_det, roll, pitch, yaw = head_pose_estimator.get_pose(frame=frame, landmarks=landmarks, frame_size=frame_size)
            asleep, looking_away, distracted = scorer.eval_scores(t_now=t_now, ear_score=ear,
                                                                  gaze_score=gaze, head_roll=roll,
                                                                  head_pitch=pitch, head_yaw=yaw)

            if frame_det is not None:
                frame = frame_det

            # Check if eyes are closed
            if ear <= args.ear_thresh:
                eyes_closed_time += 1 / fps
            else:
                eyes_closed_time = 0

            # Check for asleep condition
            if eyes_closed_time >= args.ear_time_thresh:
                asleep = True
                description += "ASLEEP! "
                asleep_flag = True
                distracted_flag = False

            if distracted:
                description += "DISTRACTED! "
                distracted_flag = True
                asleep_flag = False

            # Check for proper driving state
            if not asleep and not distracted and abs(pitch) <= args.pitch_thresh and abs(yaw) <= args.yaw_thresh and abs(roll) <= args.roll_thresh:
                description = "DRIVING PROPERLY! "
                proper_driving_flag = True
                distracted_flag = False
                asleep_flag = False

            yield frame, description  # Yield both frame and description

        e2 = cv2.getTickCount()
        proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000

        if args.show_proc_time:
            print(f"Processing Time: {proc_time_frame_ms:.2f} ms")

        if description != prev_state and description.strip():
            print(description)
            prev_state = description

        i += 1

    cap.release()

def _get_landmarks(lms):
    surface = 0
    for lms0 in lms:
        landmarks = [np.array([point.x, point.y, point.z]) for point in lms0.landmark]

        landmarks = np.array(landmarks)

        landmarks[landmarks[:, 0] < 0., 0] = 0.
        landmarks[landmarks[:, 0] > 1., 0] = 1.
        landmarks[landmarks[:, 1] < 0., 1] = 0.
        landmarks[landmarks[:, 1] > 1., 1] = 1.

        dx = landmarks[:, 0].max() - landmarks[:, 0].min()
        dy = landmarks[:, 1].max() - landmarks[:, 1].min()
        new_surface = dx * dy
        if new_surface > surface:
            biggest_face = landmarks

    return biggest_face

if __name__ == "__main__":
    main()

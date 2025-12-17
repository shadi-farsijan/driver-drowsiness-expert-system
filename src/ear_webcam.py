import cv2
import numpy as np
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh

def euclidean(a, b):
    return np.linalg.norm(a - b)

def eye_aspect_ratio(eye):
    p1, p2, p3, p4, p5, p6 = eye
    return (euclidean(p2, p6) + euclidean(p3, p5)) / (2.0 * euclidean(p1, p4))

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.23
DROWSY_TIME = 1.2

cap = cv2.VideoCapture(0)
start_time = None

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        status = "No face"
        ear = 0

        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0].landmark

            def pt(i):
                return np.array([face[i].x * w, face[i].y * h])

            right_eye = np.array([pt(i) for i in RIGHT_EYE])
            left_eye = np.array([pt(i) for i in LEFT_EYE])

            ear = (eye_aspect_ratio(right_eye) + eye_aspect_ratio(left_eye)) / 2

            if ear < EAR_THRESHOLD:
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time > DROWSY_TIME:
                    status = "DROWSY"
            else:
                start_time = None
                status = "ALERT"

            for p in right_eye.astype(int):
                cv2.circle(frame, tuple(p), 2, (0,255,0), -1)
            for p in left_eye.astype(int):
                cv2.circle(frame, tuple(p), 2, (0,255,0), -1)

        cv2.putText(frame, f"EAR: {ear:.2f}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, status, (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


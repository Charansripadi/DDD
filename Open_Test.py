



# Raw Logic file runs in python interpretor

import cv2
import dlib
import numpy as np
from collections import deque
from scipy.spatial import distance as dist
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model and dlib
model = load_model("model/cnn_model_face_drowsiness.h5")
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("dlib_models/shape_predictor_68_face_landmarks_GTX.dat")

# EAR parameters
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 15
ear_counter = 0

# Smoothing CNN predictions
pred_buffer = deque(maxlen=15)

# Eye landmark indexes
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# EAR calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for face in faces:
        landmarks = landmark_predictor(gray, face)
        shape_np = np.array([[p.x, p.y] for p in landmarks.parts()])

        left_eye = shape_np[LEFT_EYE]
        right_eye = shape_np[RIGHT_EYE]
        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye landmarks
        for (x, y) in np.concatenate((left_eye, right_eye), axis=0):
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

        # EAR logic
        if ear < EAR_THRESHOLD:
            ear_counter += 1
        else:
            ear_counter = 0

        # CNN prediction (used for supportive confidence)
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_img = frame[y1:y2, x1:x2]

        try:
            face_resized = cv2.resize(face_img, (96, 96))
            face_array = img_to_array(face_resized) / 255.0
            face_input = np.expand_dims(face_array, axis=0)

            pred = model.predict(face_input, verbose=0)[0][0]
            pred_buffer.append(pred)
            avg_pred = np.mean(pred_buffer)

        except Exception:
            avg_pred = 0.0  # fallback if model input fails

        # Final logic: EAR has priority
        is_drowsy = ear_counter >= EAR_CONSEC_FRAMES
        label = "Drowsy" if is_drowsy else "Non-Drowsy"
        color = (0, 0, 255) if is_drowsy else (0, 255, 0)

        # Display on screen
        cv2.putText(frame, f"Status: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"CNN Drowsy Score: {avg_pred:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 255), 2)

        if is_drowsy:
            cv2.putText(frame, "DROWSY! Sound ON", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Face rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.imshow("Drowsiness Detection (EAR Priority)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

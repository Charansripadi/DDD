# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import dlib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.spatial import distance as dist
import base64
from collections import deque

app = Flask(__name__)

# Load models
model = load_model("model/cnn_model_face_drowsiness.h5")
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("dlib_models/shape_predictor_68_face_landmarks_GTX.dat")

EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 3
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

ear_counter = 0
pred_buffer = deque(maxlen=15)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
@app.route('/analyze', methods=['POST'])
def analyze():
    global ear_counter
    try:
        data = request.json['image']
        img_data = base64.b64decode(data.split(',')[1])
        np_img = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        if len(faces) == 0:
            ear_counter = 0  # Reset when no face
            return jsonify({
                "status": "No face",
                "ear": 0.0,
                "cnn_score": 0.0
            })

        for face in faces:
            landmarks = landmark_predictor(gray, face)
            shape_np = np.array([[p.x, p.y] for p in landmarks.parts()])

            left_eye = shape_np[LEFT_EYE]
            right_eye = shape_np[RIGHT_EYE]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

            if ear < EAR_THRESHOLD:
                ear_counter += 1
            else:
                ear_counter = 0

            (x1, y1, x2, y2) = (face.left(), face.top(), face.right(), face.bottom())
            face_img = frame[y1:y2, x1:x2]

            try:
                face_resized = cv2.resize(face_img, (96, 96))
                face_array = img_to_array(face_resized) / 255.0
                face_input = np.expand_dims(face_array, axis=0)
                pred = model.predict(face_input, verbose=0)[0][0]
                pred_buffer.append(pred)
                avg_pred = np.mean(pred_buffer)
                print(avg_pred)
            except:
                avg_pred = 0.0

            is_drowsy = ear_counter >= EAR_CONSEC_FRAMES or avg_pred < 0.1
            print(ear_counter,EAR_CONSEC_FRAMES)
            label = "Drowsy" if is_drowsy else "Non-Drowsy"

            return jsonify({
                "status": label,
                "ear": round(ear, 2),
                "cnn_score": round(float(avg_pred), 2)
            })

    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)})


if __name__ == '__main__':
    app.run(debug=True)

# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

st.title("Driver Drowsiness Detection (CNN + Eyes)")
st.write("Upload a driver's face image to detect drowsiness.")

model = load_model("cnn_model_face_drowsiness.h5")

labels = ["Drowsy", "Non-Drowsy"]

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize((96, 96))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    pred = model.predict(image)[0][0]
    label = labels[int(round(pred))]

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {pred:.2f}")

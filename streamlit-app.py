import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# App configuration
st.set_page_config(page_title="Human Face Detection", layout="centered")
st.title("🧠 Human Face Identification App")
st.write("Upload an image and the model will identify human faces.")

# ✅ Load Haar Cascade safely using OpenCV built-in path
cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

# Safety check
if face_cascade.empty():
    st.error("❌ Failed to load Haar Cascade model.")
    st.stop()

# Upload Image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Display original image
    st.subheader("📷 Uploaded Image")
    st.image(image, width=500)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Draw rectangles and labels
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img_array,
            "Human face identified",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Display result
    st.subheader("✅ Face Detection Result")
    st.image(img_array, width=500)

    st.success(f"Total faces detected: {len(faces)}")

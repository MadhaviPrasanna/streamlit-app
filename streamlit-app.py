import streamlit as st
import cv2
import numpy as np
from PIL import Image

# App Title
st.set_page_config(page_title="Human Face Detection", layout="centered")
st.title("🧠 Human Face Identification App")
st.write("Upload an image and the model will identify human faces.")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Upload Image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Display Original Image
    st.subheader("📷 Uploaded Image")
    st.image(image, use_column_width=True)

    # Convert to Grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Detect Faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Draw Rectangles
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

    # Display Result
    st.subheader("✅ Face Detection Result")
    st.image(img_array, use_column_width=True)

    st.success(f"Total faces detected: {len(faces)}")

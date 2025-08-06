import streamlit as st
import cv2
import os
import torch
from PIL import Image
from datetime import datetime
import numpy as np
from data.Utils import load_model, resize_image, get_embedding, is_same_person, detect_faces, preprocess_image

# Ensure required folders exist
os.makedirs("./faces", exist_ok=True)

# Load model
@st.cache_resource
def get_model():
    return load_model("./model_artifacts/arcface_model.pth")

model = get_model()

st.title("🧠 Face Verification")
st.header("📸 Upload Two Images for Verification")

img1 = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"])
img2 = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"])

if img1 and img2:
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption="Image 1", width=200)
    with col2:
        st.image(img2, caption="Image 2", width=200)

    # Load and preprocess Image 1
    img1 = Image.open(img1).convert('RGB')
    img1_array = np.array(img1)
    det_img1 = detect_faces(img1_array)
    if det_img1 is None:
        st.error("No face detected in Image 1.")
        st.stop()
    image1 = resize_image(img1_array)

    # Load and preprocess Image 2
    img2 = Image.open(img2).convert('RGB')
    img2_array = np.array(img2)
    det_img2 = detect_faces(img2_array)
    if det_img2 is None:
        st.error("No face detected in Image 2.")
        st.stop()
    image2 = resize_image(img2_array)

    # Save detected faces with unique timestamps
    # ts1 = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # path1 = f"./faces/face_{ts1}.jpg"
    # cv2.imwrite(path1, cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))
    # st.info(f"Saved: {path1}")

    # ts2 = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # path2 = f"./faces/face_{ts2}.jpg"
    # cv2.imwrite(path2, cv2.cvtColor(image2, cv2.COLOR_RGB2BGR))
    # st.info(f"Saved: {path2}")

    # Generate embeddings
    emb1 = get_embedding(model, image1)
    emb2 = get_embedding(model, image2)

    # Compare embeddings
    similarity, result = is_same_person(emb1, emb2, threshold=0.5)

    st.write(f"**Cosine Similarity:** {similarity:.4f}")
    if result:
        st.success("✅ Same Person")
    else:
        st.error("❌ Different Persons")

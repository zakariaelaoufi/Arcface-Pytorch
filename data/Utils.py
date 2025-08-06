import torch
import torch.nn.functional as F
from models.FaceNet import FaceNet
import cv2
import pandas as pd
from datetime import datetime
import numpy as np
import json
import os

def load_model(model_path="./model_artifacts/arcface_model.pth"):
    model = FaceNet(num_classes=540, embedding_dim=512).to('cpu')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(model)
    model.eval()
    return model



def resize_image(image, dsize=(224, 224)):
    resized_image = cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_LANCZOS4)
    return resized_image


def generate_vggface_df(dir):
  image_path = []
  image_label = []
  for folder in os.listdir(dir):
      for label in os.listdir(dir + "/" + folder):
          for image in os.listdir(dir + "/" + folder + "/" + label):
              curr_path = dir + "/" + folder + "/" + label + "/" + image
              image_path.append(curr_path)
              image_label.append(label)

  return pd.DataFrame(zip(image_path, image_label), columns = ['image_path', 'label'])


def get_embedding(model, image):
    image = resize_image(image)

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    if image.dtype != torch.float32:
        image = image.float()

    if image.max() > 1.0:
        image = image / 255.0

    if image.ndim == 3 and image.shape[2] == 3:
        image = image.permute(2, 0, 1)

    if image.ndim == 3:
        image = image.unsqueeze(0)

    image = image.to(next(model.parameters()).device)

    model.eval()
    with torch.no_grad():
        embedding = model(image)

    return embedding.squeeze().cpu()


def average_euclidean_distance(x1, x2):
    distances = torch.norm(x1 - x2, dim=1, p=2)
    return distances.mean()


def get_cosine_sim(model, img1, img2):
    emb1, emb2 = get_embedding(model, img1), get_embedding(model, img2)
    cos_sim = F.cosine_similarity(emb1, emb2)
    return cos_sim.mean().item()


def is_same_person(emb1, emb2, threshold=0.5):
    if emb1.ndim == 1:
        emb1 = emb1.unsqueeze(0)
    if emb2.ndim == 1:
        emb2 = emb2.unsqueeze(0)
    
    similarity = F.cosine_similarity(emb1, emb2, dim=1)
    avg_sim = similarity.mean().item()
    return avg_sim, avg_sim >= threshold


def preprocess_image(image, clip_limit=2.0):
    # Convert to YCrCb color space for better contrast enhancement
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_channel_stretched = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX)

    # # Apply CLAHE (Adaptive Histogram Equalization) on the Y (luminance) channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(3, 3))
    y_clahe = clahe.apply(y_channel_stretched)

    # # Merge and convert back to RGB
    image = cv2.merge([y_clahe, cr, cb])
    image = cv2.cvtColor(image, cv2.COLOR_YCR_CB2RGB)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # Aplly sharpening
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)

    return sharpened


def detect_faces(frame):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=8)
    if len(faces) == 0:
        return None
    for idx, (x, y, w, h) in enumerate(faces):
        margin = 30
        x1 = max(x - margin, 0)
        y1 = max(y - margin - 60, 0)
        x2 = min(x + w + margin, frame.shape[1])
        y2 = min(y + h + 60 + margin, frame.shape[0])
        face_roi = frame[y1:y2, x1:x2]
    return face_roi


def generate_embedding_json(model, image, name):
    # Preprocess and get embedding
    image = resize_image(image)
    embedding = get_embedding(model, np.array(image))
    embedding_list = embedding.view(-1).tolist()

    # Create the JSON-serializable dict
    entry = {
        "name": name,
        "embedding": embedding_list,
        "timestamp": datetime.now().isoformat()
    }
    
    return entry


def process_image_folder(model, folder_path="./test", output_path="embeddings.json"):
    data = []

    for file in os.listdir(folder_path):
        if file.endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(file)[0]
            img_path = os.path.join(folder_path, file)
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            entry = generate_embedding_json(model, image, name)
            data.append(entry)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved {len(data)} embeddings to {output_path}")
import torch
import torch.nn.functional as F
import cv2
import pandas as pd
import os

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


def generate_perona_emnedding(model, image):
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
    emb1, emb2 = generate_perona_emnedding(model, img1)
    cos_sim = F.cosine_similarity(emb1, emb2)
    return cos_sim.mean().item()
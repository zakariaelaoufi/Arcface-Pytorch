import torch
from torch.utils.data import DataLoader
from data.Dataset import customDatasets, transform_augmented, transformer
from sklearn.model_selection import train_test_split
from data.Utils import generate_vggface_df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

## Note
# !pip install opendatasets --quiet
# import opendatasets as od
# od.download('https://www.kaggle.com/datasets/hearfool/vggface2')


path_data = '/content/vggface2'
BATCH_SIZE = 128


train_df = generate_vggface_df(path_data)


class_idx = {}
for i, label in enumerate(sorted(train_df['label'].unique())):
    class_idx[label] = i

train_df['label_'] = train_df['label'].map(class_idx)

train_df, val_df = train_test_split(train_df, test_size=0.12, stratify=train_df['label_'], random_state=42)
val_df, test_df = train_test_split(val_df, test_size=0.4, stratify=val_df['label_'], random_state=42)


original_train_dataset = customDatasets(train_df, transform=transformer)
test_dataset = customDatasets(test_df, transform=transformer)
val_dataset = customDatasets(val_df, transform=transformer)
augmented_train_dataset = customDatasets(train_df.sample(frac=0.75), transform=transform_augmented)

train_dataset = torch.utils.data.ConcatDataset([original_train_dataset, augmented_train_dataset])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
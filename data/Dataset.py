import torch
from torch.utils.data import Dataset
from torchvision import transforms
from data.Utils import resize_image
import cv2

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

transform_augmented = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

class customDatasets(Dataset):
  def __init__(self, dataframe, transform=None):
     super().__init__()
     self.dataframe = dataframe
     self.transform = transform

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, index):
    image_path = self.dataframe.iloc[index, 0]
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Image not found or unreadable at path: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image_resized = resize_image(image)

    label = torch.tensor(int(self.dataframe.iloc[index, 2]), dtype=torch.long)

    if self.transform:
        image_resized = self.transform(image_resized)

    return image_resized, label

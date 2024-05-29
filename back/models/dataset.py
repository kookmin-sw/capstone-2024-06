from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import requests
from io import BytesIO


class DesignImageDataset(Dataset):
    def __init__(self, npy_file, src_urls):
        self.vectors = np.load(npy_file)
        self.src_urls = src_urls
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.src_urls)

    def __getitem__(self, idx):
        response = requests.get(self.src_urls[idx] + "w=224&h=224")
        img = Image.open(BytesIO(response.content))
        vector = self.vectors[idx]
        if self.transform:
            img = self.transform(img)

        return img, vector
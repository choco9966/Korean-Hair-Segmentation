import cv2
import numpy as np
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform


    def __getitem__(self, idx):
        images = cv2.imread(self.data.loc[idx]['images'])
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        masks = cv2.imread(self.data.loc[idx]['masks'])[:,:,0]
        masks = masks.astype(float)
        masks = np.expand_dims(masks, axis=2)

        if self.transform is not None:
            transformed = self.transform(image=images, mask=masks)
            images = transformed['image']
            masks = transformed['mask']

        return images, masks

    def __len__(self):
        return len(self.data)


class ValidDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        ids = self.data.loc[idx]['images']
        images = cv2.imread(self.data.loc[idx]['images'])
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        masks = cv2.imread(self.data.loc[idx]['masks'])[:,:,0]
        masks = masks.astype(float)
        masks = np.expand_dims(masks, axis=2)

        if self.transform is not None:
            transformed = self.transform(image=images, mask=masks)
            images = transformed["image"]
            masks = transformed["mask"]

        return ids, images, masks

    def __len__(self):
        return len(self.data)


class TestDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform


    def __getitem__(self, idx):
        ids = self.data.loc[idx]['ids']
        images = cv2.imread(self.data.loc[idx]['images'])
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=images)
            images = transformed["image"]

        return ids, images

    def __len__(self):
        return len(self.data)

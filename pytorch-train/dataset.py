import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BinaryImageDatasetLoadMemory(Dataset):
    def __init__(self, file_list, transform=None, flip=False, rotate=False, divide_by=1):
        new_images = []
        new_labels = []
        for img_path in tqdm(file_list, desc="Load files"):
            image = np.load(img_path[0])/divide_by
            if transform:
                image = transform(image)
            new_images.append(np.array(image))
            new_labels.append(int(img_path[1]))
            if rotate:
                for k in [1,2,3]:
                    new_images.append(np.rot90(np.array(image), k=k ,axes=(1,2)))
                    new_labels.append(int(img_path[1]))
            if flip:
                new_images.append(np.flip(np.array(image), axis=2))
                new_labels.append(int(img_path[1]))
                if rotate:
                    for k in [1,2,3]:
                        new_images.append(np.rot90(np.flip(np.array(image), axis=2), k=k ,axes=(1,2)))
                        new_labels.append(int(img_path[1]))
        new_images = np.array(new_images)
        self.images = torch.tensor(new_images, dtype=torch.float32)
        self.labels = torch.tensor(new_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]




class BinaryImageDatasetGetIndividual(Dataset):
    def __init__(self, file_list, transform=None, flip=False, rotate=False, divide_by=1):
        new_images = []
        new_labels = []
        new_transforms = []
        self.divide_by = divide_by
        for img_path in tqdm(file_list, desc="Load files"):
            # image = np.load(img_path[0])/255
            new_images.append(img_path[0])
            new_labels.append(int(img_path[1]))
            new_transforms.append((0,0))
            if rotate:
                for k in [1,2,3]:
                    new_images.append(img_path[0])
                    new_labels.append(int(img_path[1]))
                    new_transforms.append((0,k))
            if flip:
                new_images.append(img_path[0])
                new_labels.append(int(img_path[1]))
                new_transforms.append((1,0))
                if rotate:
                    for k in [1,2,3]:
                        new_images.append(img_path[0])
                        new_labels.append(int(img_path[1]))
                        new_transforms.append((1,k))
        self.images = new_images
        self.labels = torch.tensor(new_labels, dtype=torch.float32)
        self.transforms = new_transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = np.load(self.images[idx])/self.divide_by
        # print(np.shape(image))
        if self.transforms[idx][0] == 1:
            image = np.flip(np.array(image), axis=1)
        if self.transforms[idx][1] > 0:

            image = np.rot90(image.copy(), k=self.transforms[idx][1], axes=(0,1))
        image = np.transpose(image, axes=[2,0,1])
        return torch.tensor(image.copy(), dtype=torch.float32), self.labels[idx]

transform = transforms.Compose([
    transforms.ToTensor(),
])
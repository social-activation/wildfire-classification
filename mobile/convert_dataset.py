import os
import numpy as np
from torchvision import transforms

from dataset import *



save_folders = ["wildfire_dataset/train/fire", "wildfire_dataset/train/nofire", "wildfire_dataset/test/fire", "wildfire_dataset/test/nofire"]

paths_train_fire = [save_folders[0]+"/"+img for img in os.listdir(save_folders[0])]
paths_train_nofire = [save_folders[1]+"/"+img for img in os.listdir(save_folders[1])]
paths_test_fire = [save_folders[2]+"/"+img for img in os.listdir(save_folders[2])]
paths_test_nofire = [save_folders[3]+"/"+img for img in os.listdir(save_folders[3])]

len_train = np.amin([len(paths_train_fire), len(paths_train_nofire)])
len_test = np.amin([len(paths_test_fire), len(paths_test_nofire)])

paths_train = np.append([(path, 1) for path in paths_train_fire[:len_train]], [(path, 0) for path in paths_train_nofire[:len_train]], axis=0)
paths_test = np.append([(path, 1) for path in paths_test_fire[:len_test]], [(path, 0) for path in paths_test_nofire[:len_test]], axis=0)

transform = transforms.Compose([
    # transforms.Resize((128, 128)),
    # transforms.ToPILImage(),
    # transforms.Grayscale(num_output_channels=1),
    # transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Resize((48, 48)),
    # transforms.Resize((32, 32)),
    
])

train_dataset = BinaryImageDatasetLoadMemory(paths_train, transform=transform, flip=True, rotate=True, divide_by=255)
test_dataset = BinaryImageDatasetLoadMemory(paths_test, transform=transform, divide_by=255)



import os
from torchvision.transforms.functional import to_pil_image

# Create directories to save images
os.makedirs("wildfire_dataset/png/train", exist_ok=True)
os.makedirs("wildfire_dataset/png/test", exist_ok=True)

# Save train dataset images
for idx, (img, label) in enumerate(train_dataset):
    img_pil = to_pil_image(img)  # Convert tensor to PIL image
    img_pil.save(f"wildfire_dataset/png/train/image_{idx:04d}_label_{label}.png")

# Save test dataset images
for idx, (img, label) in enumerate(test_dataset):
    img_pil = to_pil_image(img)
    img_pil.save(f"wildfire_dataset/png/test/image_{idx:04d}_label_{label}.png")

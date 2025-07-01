import ezkl
import json
import os

import numpy as np

import torch
from torchvision import transforms
import asyncio
from tqdm import tqdm

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

test_dataset = BinaryImageDatasetLoadMemory(paths_test, transform=transform, divide_by=255)



def load_json(file_name):
    with open(file_name) as f:
        output = json.load(f)
    return output

# Output
def save_json(file_name, data):
    with open(file_name,'w') as f:
        json.dump(data, f)

os.makedirs("wildfire_dataset/zkp_compatible", exist_ok=True)

answer_section = {}
for i, (img, label) in tqdm(enumerate(test_dataset), desc="Convert fire data into json format"):
    file_name = str(i).zfill(4)

    x = img.cpu().detach().numpy().reshape([-1]).tolist()

    save_json("wildfire_dataset/zkp_compatible/"+file_name+".json", {"input_data": [x]})

    answer_section.update({file_name: int(label.numpy())})

save_json("wildfire_dataset/zkp_compatible_dataset.json", answer_section)
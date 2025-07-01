import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import confusion_matrix, accuracy_score

from dataset import *
from models import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = LeNet().to(device)
# model = SmallLeNet().to(device)
model = LargerFeatureExtractorNet().to(device)

# Load saved weights
model.load_state_dict(torch.load("training_data/wildfire_classifier.pth", map_location=torch.device(device)))


model.eval()
all_preds = []
all_labels = []

batch_size = 64

save_folders = ["wildfire_dataset/train/fire", "wildfire_dataset/train/nofire", "wildfire_dataset/test/fire", "wildfire_dataset/test/nofire"]

paths_test_fire = [save_folders[2]+"/"+img for img in os.listdir(save_folders[2])]
paths_test_nofire = [save_folders[3]+"/"+img for img in os.listdir(save_folders[3])]

len_test = np.amin([len(paths_test_fire), len(paths_test_nofire)])

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

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = (outputs > 0.5).float().squeeze()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds).astype(int)
all_labels = np.array(all_labels).astype(int)

confu = confusion_matrix(all_labels, all_preds, labels=[1, 0])

fig, ax = plt.subplots()
im = ax.imshow(confu, cmap='Blues')

# Set axis labels
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')

# Show all ticks and label them
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Positive', 'Negative'])
ax.set_yticklabels(['Positive', 'Negative'])

for i in range(confu.shape[0]):
    for j in range(confu.shape[1]):
        ax.text(j, i, confu[i, j], ha='center', va='center', color='black', fontsize=12)


acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {acc:.5f}")

plt.savefig("training_data/confusion_matrix.png")
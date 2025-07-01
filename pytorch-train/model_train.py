import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import accuracy_score

from dataset import *
from models import *



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

print("Training set:", len(paths_train), "| Testing set:", len(paths_test))





num = 4

indices = random.sample(range(len(train_dataset)), num)

plt.clf()
fig, axes = plt.subplots(1,num, figsize=(10,3), layout="constrained")

for ax, idx in zip(axes, indices):
    image, label = train_dataset[idx]
    image = image.permute(1, 2, 0)
    ax.imshow(image)
    ax.set_title(f"Label: {label}")
    ax.axis('off')

plt.savefig("training_data/display_model_input.png")







batch_size=64
num_epochs=100
lr=0.001

img, label = train_dataset[0]
print(img.shape)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# model = LeNet().to(device)
model = SmallLeNet().to(device)
criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

model.train()
description = f"Loss: -.------- - Accuracy: -.---"
pbar = tqdm(range(num_epochs), desc=description)
loss_track = []
acc_track = []

acc = 0

for epoch in pbar:
    running_loss = 0.0
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape (batch, 1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    loss_track.append(running_loss/len(train_dataloader))

    if epoch == 0 or epoch % 10 == 0 or epoch == num_epochs-1:
        model.eval()
        all_preds = []
        all_labels = []

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                # print(outputs)
                preds = (outputs > 0.5).float().squeeze()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        acc_track.append(acc)

        model.train()

    pbar.set_description(f"Loss: {running_loss/len(train_dataloader):.10f} - Accuracy: {acc:.3f}")

fig, axs = plt.subplots(2,1,figsize=(4.5,9), layout="constrained")

axs[0].plot(loss_track)
axs[0].set_title("Loss")
axs[0].set_yscale("log")

axs[1].plot(acc_track)
axs[1].set_title("Accuracy")

plt.savefig("training_data/training_loss_and_accuracy.png")


torch.save(model.state_dict(), "training_data/wildfire_classifier.pth")
import kagglehub
import os
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image



# Download latest version
path = kagglehub.dataset_download("elmadafri/the-wildfire-dataset")

print("Path to dataset files:", path)

print(os.listdir(path))





local_path = path
train_files_fire = np.array([])
train_files_no_fire = np.array([])
for addition in ["/the_wildfire_dataset_2n_version/train/fire/"]:
    just_file_name = os.listdir(local_path+addition)
    train_files_fire = np.append(train_files_fire, [local_path+addition+img for img in just_file_name])
for addition in ["/the_wildfire_dataset_2n_version/train/nofire/"]:
    just_file_name = os.listdir(local_path+addition)
    train_files_no_fire = np.append(train_files_no_fire, [local_path+addition+img for img in just_file_name])

print("Train: ", len(train_files_fire), len(train_files_no_fire))


local_path = path
test_files_fire = np.array([])
test_files_no_fire = np.array([])
for addition in ["/the_wildfire_dataset_2n_version/test/fire/"]:
    just_file_name = os.listdir(local_path+addition)
    test_files_fire = np.append(test_files_fire, [local_path+addition+img for img in just_file_name])
for addition in ["/the_wildfire_dataset_2n_version/test/nofire/"]:
    just_file_name = os.listdir(local_path+addition)
    test_files_no_fire = np.append(test_files_no_fire, [local_path+addition+img for img in just_file_name])

print("Test: ", len(test_files_fire), len(test_files_no_fire))





class ToNumpy:
    def __call__(self, image):
        return np.array(image)

transform = transforms.Compose([
    transforms.Resize((256,256)),
    ToNumpy()
])

save_folders = ["wildfire_dataset/train/fire", "wildfire_dataset/train/nofire", "wildfire_dataset/test/fire", "wildfire_dataset/test/nofire"]
og_files = [train_files_fire, train_files_no_fire, test_files_fire, test_files_no_fire]

for folder, file_list in zip(save_folders, og_files):
    print(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i, img_path in tqdm(enumerate(file_list), desc="Convert to 256x256 np files", total=len(file_list)):
        if not os.path.exists(folder+"/"+str(i).zfill(4)):
            image = Image.open(img_path).convert("RGB")
            image = transform(image)
            np.save(folder+"/"+str(i).zfill(4),image)
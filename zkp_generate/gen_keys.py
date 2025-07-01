import ezkl
import json
import os

import numpy as np

import torch
from torchvision import transforms
import asyncio

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

async def model_to_zkp_generator(save_directory: str, calibration_scale, calibration_data_points: int = 100):

    os.makedirs(save_directory, exist_ok=True)

    # Define paths
    model_path = "training_data/model.onnx"
    compiled_model_path = os.path.join(save_directory, "network.compiled")
    pk_path = os.path.join(save_directory, "key.pk")
    vk_path = os.path.join(save_directory, "key.vk")
    settings_path = os.path.join(save_directory, "settings.json")
    cal_path = os.path.join(save_directory, "cal_data.json")
    abi_path = os.path.join(save_directory, "test.abi")
    sol_code_path = os.path.join(save_directory, "test_1.sol")

    # Run settings
    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = "private"
    run_args.param_visibility = "private"   # "fixed"
    run_args.output_visibility = "public"
    run_args.num_inner_cols = 2
    run_args.variables = [("batch_size", 1)]

    num_data_points = 8

    # Fetch 30 data points from the train_dataset
    data_points = []
    for i, (data_point, _) in enumerate(train_dataset):
        if i >= num_data_points:
            break
        data_points.append(data_point)

    # # Prepare calibration data
    # data_points = [data for i, (data, _) in enumerate(train_dataset) if i < calibration_data_points]
    train_data_batch = torch.stack(data_points)

    if train_data_batch.dim() == 3:
        train_data_batch = train_data_batch.unsqueeze(0)

    x = train_data_batch.cpu().numpy().reshape([-1]).tolist()
    json.dump({"input_data": [x]}, open(cal_path, 'w'))

    print("Creating settings")
    # Generate settings
    assert ezkl.gen_settings(model_path, settings_path, py_run_args=run_args)

    # Calibration
    print("Calibrating")
    # def run_calibration():
    await ezkl.calibrate_settings(cal_path, model_path, settings_path, "resources", scales=[calibration_scale])
    # await run_calibration()

    # Compile circuit
    print("Compiling Circuit")
    assert ezkl.compile_circuit(model_path, compiled_model_path, settings_path)

    print("Getting SRS")
    await ezkl.get_srs(settings_path)

    # # Setup SRS
    # print("Getting SRS")
    # async def run_get_srs():
    #     assert await ezkl.get_srs(settings_path)
    # run_get_srs()

    print("Setting up proving and verifying keys")
    # Setup proving/verifying keys
    # assert ezkl.setup(compiled_model_path, vk_path, pk_path)
    res = ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
    )
    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)

    # Create EVM verifier
    print("Running evm verifier")
    async def run_create_evm_verifier():
        assert await ezkl.create_evm_verifier(vk_path, settings_path, sol_code_path, abi_path)
    run_create_evm_verifier()


for x in [10]:
    for y in [3,8]:
        directory = f"zkp_data/{x}_{y}"
        # if not os.path.exists(directory):
        asyncio.run(model_to_zkp_generator(directory, y, x))

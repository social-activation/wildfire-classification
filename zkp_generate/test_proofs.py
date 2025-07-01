import os
from tqdm import tqdm
import numpy as np

import nest_asyncio

import ezkl
import os
import torch
from torchvision import datasets, transforms
import json
import asyncio
import nest_asyncio
from pytictoc import TicToc

def load_json(file_name):
    with open(file_name) as f:
        output = json.load(f)
    return output

async def infer_and_build_proof(compiled_model_path, settings_path, pk_path, vk_path, input_path, output_path):

    witness_path = "witness.json"

    t = TicToc()

    t.tic()
    await ezkl.gen_witness(input_path, compiled_model_path, witness_path)
    elapsed_gen_witness = t.tocvalue()

    t.tic()
    # GENERATE A PROOF
    res = ezkl.prove(
            witness_path,
            compiled_model_path,
            pk_path,
            output_path,
            "single",
        )
    elapsed_prove = t.tocvalue()

    print("Time for Gen Witness:", elapsed_gen_witness)
    print("Time for Prove:", elapsed_prove)

    assert os.path.isfile(output_path)
    os.remove(witness_path)




how_many_to_test = 100

indices = np.array(np.append(np.arange(0,50), np.arange(200,250)), dtype=int)

test_dataset_files = np.array(list(load_json("wildfire_dataset/zkp_compatible_dataset.json").keys()))[indices]
test_dataset_paths = ["wildfire_dataset/zkp_compatible/"+file+".json" for file in test_dataset_files]

# print(test_dataset_files)

def gen_proof(zkp_generator_path):
    compiled_model_path = os.path.join(zkp_generator_path, "network.compiled")
    settings_path = os.path.join(zkp_generator_path, "settings.json")
    pk_path = os.path.join(zkp_generator_path, "key.pk")
    vk_path = os.path.join(zkp_generator_path, "key.vk")

    os.makedirs(os.path.join(zkp_generator_path, "proofs"), exist_ok=True)

    for file_name, file_path in tqdm(zip(test_dataset_files, test_dataset_paths), desc="Creating proofs", total=len(test_dataset_files)):

        proof_path = os.path.join(zkp_generator_path, "proofs", file_name+"_proof.pf")
        # nest_asyncio.apply()

        # print()
        # print(np.shape(load_json(file_path)["input_data"]))
        # asyncio.run(ezkl.gen_witness(file_path, compiled_model_path, "witness.json"))
        # break
        if not os.path.exists(proof_path):
            asyncio.run(infer_and_build_proof(compiled_model_path, settings_path, pk_path, vk_path, file_path, proof_path))
        # break



for x in [10]:
    for y in [3,8]:
        save_directory = os.path.join("zkp_data", f"{x}_{y}")
        gen_proof(save_directory)








correct_answers = load_json("wildfire_dataset/zkp_compatible_dataset.json")

accuracy = 0
count = 0
for file_name in tqdm(test_dataset_files, desc="Creating proofs", total=len(test_dataset_files)):
    file_path = "zkp_data/10_3/proofs/"+file_name+"_proof.pf"
    proof_data = load_json(file_path)
    if int(round(float(proof_data["pretty_public_inputs"]["rescaled_outputs"][0][0]))) == correct_answers[file_name]:
        accuracy += 1
    count += 1
print()
print("Proof accuracies log 3:", 100*accuracy/count, "%")


accuracy = 0
count = 0
for file_name in tqdm(test_dataset_files, desc="Creating proofs", total=len(test_dataset_files)):
    file_path = "zkp_data/10_8/proofs/"+file_name+"_proof.pf"
    proof_data = load_json(file_path)
    if int(round(float(proof_data["pretty_public_inputs"]["rescaled_outputs"][0][0]))) == correct_answers[file_name]:
        accuracy += 1
    count += 1
print()
print("Proof accuracies log 8:", 100*accuracy/count, "%")
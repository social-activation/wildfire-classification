import torch
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = LeNet().to(device)
model = SmallLeNet().to(device)
# model = LargerFeatureExtractorNet().to(device)

# Load saved weights
model.load_state_dict(torch.load("training_data/wildfire_classifier.pth", map_location=torch.device(device)))

model.eval()

sample_inputs = (torch.randn(1, 3, 48, 48), )

et_program = to_edge_transform_and_lower(
    torch.export.export(model, sample_inputs),
    partitioner=[XnnpackPartitioner()]
).to_executorch()


torch.jit.save("mobile_data/model.pte")
# with open("mobile_data/model.pte", "wb") as f:
#     f.write(et_program.buffer)




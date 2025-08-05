import torch
from executorch.runtime import Runtime
from typing import List


from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = LeNet().to(device)
model = SmallLeNet().to(device)
# model = LargerFeatureExtractorNet().to(device)

# Load saved weights
model.load_state_dict(torch.load("training_data/wildfire_classifier.pth", map_location=torch.device(device)))

model.eval()

runtime = Runtime.get()

input_tensor: torch.Tensor = torch.randn(1, 3, 48, 48)
program = runtime.load_program("mobile_data/model.pte")
method = program.load_method("forward")
output: List[torch.Tensor] = method.execute([input_tensor])
print("Run succesfully via executorch")


# eager_reference_model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
model = model(input_tensor)

print("Comparing against original PyTorch module")
print(torch.allclose(output[0], model, rtol=1e-3, atol=1e-5))
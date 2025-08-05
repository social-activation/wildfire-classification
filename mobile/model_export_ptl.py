import torch
from models import SmallLeNet  # your model definition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate and load your model
model = SmallLeNet().to(device)
model.load_state_dict(torch.load("training_data/wildfire_classifier.pth", map_location=device))
model.eval()

# Prepare example input with expected shape
example_input = torch.randn(1, 3, 48, 48).to(device)

# Export to TorchScript using tracing
traced_script_module = torch.jit.script(model)
# traced_script_module = torch.jit.trace(model, example_input)

# scripted_model = torch.jit.script(model)  # or .trace(...)

# Save the model for mobile (this strips Python bytecode)
traced_script_module._save_for_lite_interpreter("mobile_data/model.torchscript.ptl")

# Save the TorchScript model for Android (.pt or .ptl extension)
traced_script_module.save("mobile_data/model.ptl")

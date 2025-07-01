import torch
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = LeNet().to(device)
model = SmallLeNet().to(device)
# model = LargerFeatureExtractorNet().to(device)

# Load saved weights
model.load_state_dict(torch.load("training_data/wildfire_classifier.pth", map_location=torch.device(device)))

def export_model_to_onnx(model, filename="fire_classifier.onnx"):
    model.eval()
    model.cpu()

    # Create a dummy input matching (batch_size, channels, height, width)
    # dummy_input = torch.randn(1, 3, 32, 32).to(device)  # 10-band, resized to 64x64
    dummy_input = torch.randn(1, 3, 48, 48)  # 10-band, resized to 64x64
    # dummy_input = torch.randn(1, 3, 64, 64).to(device)  # 10-band, resized to 64x64
    # dummy_input = torch.randn(1, 3, 128, 128).to(device)  # 10-band, resized to 64x64
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        input_names=['input'],
        output_names=['output'],
        opset_version=13,
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"Model successfully exported to: {filename}")

export_model_to_onnx(model, "training_data/model.onnx")
import torch
from PIL import Image
from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_Weights

def run_inference(model: FasterRCNN, image: Image, device):
    transform = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()
    img_tensor = transform(image).to(device)

    with torch.no_grad():
        output = model([img_tensor])
    
    return output

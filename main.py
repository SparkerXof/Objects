import torch
import tkinter as tk
from PIL import Image
from app.app import App
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from detector.inference import run_inference
from detector.model import get_model

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model().to(device)
    model.load_state_dict(torch.load('./trained.pt', weights_only=True, map_location=device))
    model.eval()

    #root = tk.Tk()
    #app = App(root)
    #root.mainloop()

    img = Image.open("/home/sparkerxof/MTUCI/Objects/test.jpg")

    res = run_inference(model, img, device)
    print(res)

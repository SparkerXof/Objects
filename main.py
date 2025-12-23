import torch
import tkinter as tk
from app.app import App
from detector.model import get_model

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model().to(device)
    model.load_state_dict(torch.load('./detector/trained.pt', weights_only=True, map_location=device))
    model.eval()

    root = tk.Tk()
    app = App(root, model, device)
    root.mainloop()

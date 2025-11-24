import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

from engine import train_one_epoch, evaluate
import utils

from dataset import ObjectDataset
from model import get_model

TRAIN_PATH = "./data/train/"
TEST_PATH = "./data/test/"

def train(model):
    train_data = ObjectDataset(TRAIN_PATH, 480, 480, FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms())
    test_data = ObjectDataset(TRAIN_PATH, 480, 480, FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms())
    
    data_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    for epoch in range(10):
        print(f"Epoch {epoch}")
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, test_data_loader, device=device)

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = get_model().to(device)
    model.train()
    train(model)

    torch.save(model.state_dict(), './trained.pt')

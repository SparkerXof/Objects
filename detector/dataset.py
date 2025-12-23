import os, torch
import cv2 as cv
import numpy as np
from PIL import Image

class ObjectDataset(torch.utils.data.Dataset):
    def __init__(self, files_dir, width, height, transforms=None):
        self.transforms = transforms
        self.image_dir = os.path.join(files_dir, "images/")
        self.label_dir = os.path.join(files_dir, "labels/")
        self.height = height
        self.width = width

        self.images = [image for image in sorted(os.listdir(self.image_dir)) if image[-4:]=='.jpg']
        self.labels = [label for label in sorted(os.listdir(self.label_dir)) if label[-4:]=='.txt']

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        img = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
        img_res = cv.resize(img, (self.width, self.height), cv.INTER_AREA)
        img_res = Image.fromarray(img_res)

        annot_path = os.path.join(self.label_dir, self.labels[idx])
        img_boxes = []
        img_labels = []
        
        
        with open(annot_path, 'r') as file:
            for annot in file.read().splitlines():
                strdata = annot.split()
                data = np.array(strdata, dtype=np.float64)
                if len(data) > 5:
                    x_cords = data[1:-1:2]
                    y_cords = data[2:-2:2]
                    x_min = min(x_cords)
                    x_max = max(x_cords)
                    y_min = min(y_cords)
                    y_max = max(y_cords)
                else:
                    x_center = float(data[1]) * self.width
                    y_center = float(data[2]) * self.height
                    box_width = float(data[-2]) * self.width
                    box_height = float(data[-1]) * self.height
                    x_min = x_center - (box_width / 2)
                    x_max = x_center + (box_width / 2)
                    y_min = y_center - (box_height / 2)
                    y_max = y_center + (box_height / 2)

                img_labels.append(1 if (int(data[0]) == 0) else 2)
                img_boxes.append([x_min, y_min, x_max, y_max])
        
        image_id = torch.tensor([idx])
        img_labels = torch.as_tensor(img_labels, dtype=torch.int64)
        img_boxes = torch.as_tensor(img_boxes, dtype=torch.float32)
        area = (img_boxes[:, 3] - img_boxes[:, 1]) * (img_boxes[:, 2] - img_boxes[:, 0])
        iscrowd = torch.zeros((img_boxes.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = img_boxes
        target["labels"] = img_labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = idx

        if self.transforms:
            img_res = self.transforms(img_res)
        
        return img_res, target

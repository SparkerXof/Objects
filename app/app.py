import cv2 as cv
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import filedialog
from detector.inference import run_inference

class App():
    def __init__(self, root, model=None, device='cpu', *args, **kwargs):
        self.root = root

        self.model = model
        self.device = device

        self.imagepath = None
        self.image = None
        self.boxed_image = None

        self.canvas = tk.Canvas(self.root, width=640, height=480, bg = 'gray')
        self.canvas.grid(column=0, row=0)

        self.frame = tk.Frame(self.root)
        self.frame.grid(column=0, row=1)

        self.upload_button = tk.Button(self.frame, text="Upload Image", command=self.upload_image)
        self.upload_button.grid(column=0, row=0, sticky='w', padx=5, pady=5)

        self.inference_button = tk.Button(self.frame, text="Find boxes", command=self.inference_image)
        self.inference_button.grid(column=1, row=0, sticky='w', padx=5, pady=5)
    
    def draw_bboxes(self, img, bboxes, labels, scores, threshold):
        cv_img = np.array(img)[:, :, ::-1].copy()
        for i in range(len(labels)):
            if scores[i] > threshold:
                x, y, w, h = bboxes[i].astype('int')
                color = (0, 255, 0) if labels[i] == 1 else (0, 0, 255)

                cv.rectangle(cv_img, (x, y), (w, h), color, 3)
        return cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)

    def image_path(self) -> str:
        filetypes = [("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", '*.*')]
        return filedialog.askopenfilename(filetypes=filetypes)
    
    def upload_image(self):
        self.imagepath = self.image_path()
        self.image = ImageTk.PhotoImage(Image.open(self.imagepath).resize((640, 480)))
        self.canvas.create_image(1, 1, image=self.image, anchor=tk.NW)

    def inference_image(self):
        img = Image.open(self.imagepath)
        res = run_inference(self.model, img, self.device)
        self.boxed_image = Image.fromarray(self.draw_bboxes(img=img, 
                                                            bboxes=res[0]['boxes'].numpy(), 
                                                            labels=res[0]['labels'].numpy(), 
                                                            scores=res[0]['scores'].numpy(), 
                                                            threshold=0.9))
        self.boxed_image = ImageTk.PhotoImage(self.boxed_image.resize((640, 480)))
        self.canvas.create_image(1, 1, image=self.boxed_image, anchor=tk.NW)

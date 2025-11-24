import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Object on Surface Classifier")
        self.root.geometry("600x600")
        self.image_w = 400
        self.image_h = 300

        self.create_widgets()

    def create_widgets(self):
        self.image = tk.Label(
            self.root, 
            width=10,
            height=10,
            bg='gray',
        )
        self.image.pack()

        self.object_class = tk.Label(
            self.root,
            text="",
            font=('Arial', 12)
        )
        self.object_class.pack()

        self.load_image = tk.Button(
            self.root,
            text="Load an image",
            command=self.select_image,
            font=("Arial", 12),
            bg='lightgray'
        )
        self.load_image.pack()

    def select_image(self):
        filepath = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.gif *.bmp"),
                ("All files", "*.*")
            ]
        )
        if filepath:
            self.display_image(filepath)
    
    def display_image(self, filepath):
        try:
            img = Image.open(filepath)
            img.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(img)
            self.image.configure(image=photo)
            self.image.image = photo
        except:
            self.object_class.configure(text="Image load error")

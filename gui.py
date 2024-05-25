import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import torch
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'

weights = EfficientNet_V2_S_Weights.DEFAULT
data_transforms = weights.transforms()

# Load model
path = "weights_3/Epoch10_Acc0.9537.pth"
TRAIN_MODE = {"anms": 90}
num_classes = TRAIN_MODE["anms"]

# Initialize the model
model = models.efficientnet_v2_s(weights=weights).to(device)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes).to(device)

# Load state dictionary
model.load_state_dict(torch.load(path, map_location=device))
model.eval()

# Read obj_names
with open('name.txt', 'r') as file1:
    Lines = file1.readlines()

myl = [line.strip().translate({ord(ch): None for ch in '0123456789\t'}) for line in Lines]
myl.sort()

# Initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Animal Classification')
top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail((top.winfo_width() / 2.25, top.winfo_height() / 2.25))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        img = data_transforms(uploaded).to(device)
        img = img.unsqueeze(0)
        # Predict
        output = model(img)
        _, predicted = torch.max(output, 1)
        sign = "Model predicts that animal is: " + myl[predicted]
        print(sign)
        label.configure(foreground='#011638', text=sign)
    except Exception as e:
        print(f"Error: {e}")

upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Animal Classification", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()

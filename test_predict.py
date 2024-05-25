import torch
from torchvision import models
from PIL import Image
import torch.nn as nn
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
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

#Read obj_names
file1 = open('name.txt', 'r')
Lines = file1.readlines()
myl=[]
for line in Lines:
    string = line.strip().replace("\t","")
    for i in range(10):
        string = string.replace(str(i),'')
    myl.append(string)
myl.sort()
# print(myl)

#Load data
batch_size = 8
dataset_dir = "animals/animals"
testset = datasets.ImageFolder(root=dataset_dir, transform=data_transforms)
test_load = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
images, labels = next(iter(test_load))
images = images.to(device)

#Predict
outputs = model(images)
_, predicted = torch.max(outputs, 1)


# Show results
for i in range(batch_size):
    plt.subplot(2, int(batch_size/2), i + 1)
    img = images[i]
    img = img / 2 + 0.5
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    color = "green"
    label = myl[predicted[i]]
    if myl[labels[i]] != myl[predicted[i]]:
        color = "red"
        label = "(" + label + ")"
    plt.title(label, color=color)
plt.suptitle('Objects Found by Model', size=20)
plt.show()
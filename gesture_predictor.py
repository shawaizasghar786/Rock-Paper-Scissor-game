import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2

import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GestureCNN(nn.Module):
    def __init__(self):
        super(GestureCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 30 * 30, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model=GestureCNN()
model.load_state_dict(torch.load('gesture_model.pth', map_location=torch.device('cpu')))
model.eval()

labels=['rock','paper','scissors']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
def predict_gesture(frame):
    img=transform(frame)
    img= transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output=model(img)
        print("Raw output:", output)
        _,predicted=torch.max(output, 1)
    return labels[predicted.item()]


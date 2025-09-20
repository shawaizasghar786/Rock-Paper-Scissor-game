import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

Img_Size=128
Batch_Size=32
Epochs=10
Device=torch.device("cuda" if torch.cuda.is_available()else "cpu")
transform = transforms.Compose([
    transforms.Resize((Img_Size, Img_Size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
dataset=datasets.ImageFolder('dataset',transform=transform)
loader=DataLoader(dataset,batch_size=Batch_Size,shuffle=True)

class GestureCNN(nn.Module):
    def __init__(self):
        super(GestureCNN,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3,32,3),nn.Relu()
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
model = GestureCNN().to(Device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Training
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'gesture_model.pth')
print("✅ Model saved as gesture_model.pth")
    

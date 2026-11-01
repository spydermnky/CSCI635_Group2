import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import time

class Net(nn.Module):
    def __init__(self, num_classes=2):
        super(Net, self).__init__()
        # Input: [batch_size, 3, 224, 224]
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Max Pooling Layer (halves the height and width)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After three poolings, a 224x224 image becomes 28x28
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Apply Convolutions, ReLU activations, and Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the multi-dimensional tensor into a vector for the dense layers
        x = torch.flatten(x, start_dim=1)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_transform_pipeline():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
def train_model(model, dataloader, device, criterion, optimizer, num_epochs, model_path):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}')
        t1 = time.time()
        num_samples_trained = 0
        for images, targets in dataloader:
            num_samples_trained += len(images)
            print(f'Samples Trained: {num_samples_trained}')
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        t2 = time.time()
        print(f'Epoch {epoch} length: {t2-t1:0.3f}')
        torch.save(model.state_dict(), f'model_path_{epoch}')

def test_model(dataloader, device, model_path):
    model = Net()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            
            _, predictions = torch.max(outputs, dim=1)
            
            running_corrects += torch.sum(predictions == targets).item()
            total_samples += targets.size(0)
            
    accuracy = (running_corrects / total_samples) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
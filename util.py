import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Fully Connected Layers
        # After three poolings, a 224x224 image becomes 28x28
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Apply Convolutions, ReLU activations, and Pooling
        x = self.pool(F.relu(self.conv1(x)))  # Shape: [batch, 16, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # Shape: [batch, 32, 56, 56]
        x = self.pool(F.relu(self.conv3(x)))  # Shape: [batch, 64, 28, 28]
        
        # Flatten the multi-dimensional tensor into a vector for the dense layers
        x = torch.flatten(x, start_dim=1)     # Shape: [batch, 64 * 28 * 28]
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_transform_pipeline():
    return transforms.Compose([
        transforms.Resize((224, 224)),       # Resize to model input size
        transforms.ToTensor(),                # Converts PIL Image to Tensor (scales pixels to [0.0, 1.0])
        transforms.Normalize(                 # Normalize with ImageNet weights (optional but recommended)
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
def train_model(model, dataloader, device, criterion, optimizer, num_epochs, model_path):
    for epoch in range(num_epochs):
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), model_path)

def test_model(dataloader, device, model_path):
    model = Net()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    # 1. Switch model to evaluation mode
    # This disables training-specific behaviors like Dropout and Batch Normalization
    model.eval()
    
    running_corrects = 0
    total_samples = 0
    
    # 2. Disable gradient computation to optimize memory and speed
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            print(targets)
            
            # Forward pass to get raw logits
            outputs = model(images)
            
            # Convert raw logits to predicted class indices
            # torch.max returns (values, indices); we only need the indices [1]
            _, predictions = torch.max(outputs, dim=1)
            print(predictions)
            
            # Track running statistics
            running_corrects += torch.sum(predictions == targets).item()
            total_samples += targets.size(0)
            
    # 3. Calculate final metric
    accuracy = (running_corrects / total_samples) * 100
    print(running_corrects)
    print(total_samples)
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from util import Net, train_model, test_model, get_transform_pipeline

NUM_EPOCHS = 1
TRAIN_MODE = True
TEST_MODE = True
MODEL_PATH = 'trained_model.pth'

transform_pipeline = get_transform_pipeline()

train_dataset = ImageFolder(root = './SmallPetImages/', transform = transform_pipeline)
train_dataloader = DataLoader(train_dataset, batch_size = 100, shuffle = True)
test_dataset = ImageFolder(root = './SmallPetImagesTest/', transform = transform_pipeline)
test_dataloader = DataLoader(test_dataset, batch_size = 100, shuffle = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(num_classes = 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

if TRAIN_MODE: 
    train_model(model, train_dataloader, device, criterion, optimizer, NUM_EPOCHS, MODEL_PATH)
if TEST_MODE:
    test_model(train_dataloader, device, MODEL_PATH)
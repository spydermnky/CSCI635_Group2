import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from util import Net, train_model, validate_model, get_transform_pipeline, plot_accuracies

NUM_EPOCHS = 10
TRAIN_MODE = False
VALIDATION_MODE = True
TEST_MODE = False
MODEL_PATH = 'trained_model'
MODEL_FILE_TYPE = '.pth'

transform_pipeline = get_transform_pipeline()

train_dataset = ImageFolder(root = 'data/TrainingPetImages/', transform = transform_pipeline)
train_dataloader = DataLoader(train_dataset, batch_size = 10, shuffle = True)
validation_dataset = ImageFolder(root = 'data/ValidationPetImages/', transform = transform_pipeline)
validation_dataloader = DataLoader(validation_dataset, batch_size = 10, shuffle = True)
test_dataset = ImageFolder(root = 'data/TestPetImages/', transform = transform_pipeline)
test_dataloader = DataLoader(test_dataset, batch_size = 10, shuffle = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(num_classes = 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

if TRAIN_MODE: 
    train_model(model, train_dataloader, device, criterion, optimizer, NUM_EPOCHS, MODEL_PATH, MODEL_FILE_TYPE)
if VALIDATION_MODE:
    validation_accuracies = validate_model(validation_dataloader, device, NUM_EPOCHS, MODEL_PATH, MODEL_FILE_TYPE)
    train_accuracies = validate_model(train_dataloader, device, NUM_EPOCHS, MODEL_PATH, MODEL_FILE_TYPE)
    fig, ax = plot_accuracies(validation_accuracies, NUM_EPOCHS, 'Validation Accuracy')
    plot_accuracies(train_accuracies, NUM_EPOCHS, 'Training Accuracy', fig = fig, ax = ax)
if TEST_MODE:
    test_model(test_dataloader, device, NUM_EPOCHS, MODEL_PATH, MODEL_FILE_TYPE)
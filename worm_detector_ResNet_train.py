from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# num_epochs = 25  # Define the number of epochs
num_epochs = 100  # Define the number of epochs

# Define transforms
# data augmentation: resize, rotate, flip, jitter
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# basic transformations to validation set
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder('worm_categories/train', transform=train_transforms)
val_dataset = ImageFolder('worm_categories/val', transform=val_transforms)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Load a pre-trained ResNet model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Replace the last fully connected layer with a layer matching the number of categories (4 in this case)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

# Move the model to the GPU if available
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# mps does not appear to be working correctly in current version of pytorch.
# device = "cpu"

device = torch.device(device)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        # Backward + optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset)

    # At the end of each training epoch, validate the model
    model.eval()  # Set model to evaluate mode
    val_running_loss = 0.0
    val_running_corrects = 0

    # Disabling gradient calculation is important for validation.
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)

    val_epoch_loss = val_running_loss / len(val_dataset)
    val_epoch_acc = val_running_corrects / len(val_dataset)

    print(f'Epoch {epoch}/{num_epochs - 1} Train Loss:      {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    print(f'Epoch {epoch}/{num_epochs - 1} Validation Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')

now = datetime.utcnow().strftime('%Y-%m-%d_%H%M')
torch.save(model.state_dict(), f"models/model_weights_{now}.pth")
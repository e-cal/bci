import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(400, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 2)
        self.fc3 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x

net = CNN()

# building datasets

data = pd.read_csv('blink_1.csv', header=0)
data = pd.DataFrame(data)
X = data.loc[:, data.columns != '400']
y = data.loc[:, data.columns == '400']


# Define the training function
def train(model, optimizer, criterion, X, y):
    model.train()
    for inputs, labels in X, y:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Define the validation function
def validate(model, criterion, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            predicted = outputs.argmax(dim=1, keepdim=True)
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    return val_loss, accuracy

# Set up the cross-validation
kfold = KFold(n_splits=5, shuffle=True)

# Train the model using cross-validation
for fold, (train_ids, val_ids) in enumerate(kfold.split(mnist_train)):
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
    train_loader = DataLoader(mnist_train, batch_size=64, sampler=train_subsampler)
    val_loader = DataLoader(mnist_train, batch_size=64, sampler=val_subsampler)

    # Create a new model for each fold
    model = CNN()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model for several epochs
    for epoch in range(10):
        train(model, optimizer, criterion, train_loader)
        val_loss, accuracy = validate(model, criterion, val_loader)
        print(f'Fold [{fold+1}/5], Epoch [{epoch+1}/10], Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from  torchvision.datasets import ImageFolder
from torch.utils.data import SubsetRandomSampler, DataLoader
from gpu import check_gpu_availability
from ResNet import ResNet50


def preprocessData(batch_size):
    # image directory
    image_folder = '/input/COVID-19_Radiography_Dataset'

    # Set the path to your image folders
    image_path = os.getcwd() + image_folder


    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    '''
    ImageFolder will store all images in COVID folder and Normal folder into one dataset. Target 0
    is assigned to COVID images. Target 1 is assigned to Normal images.
    '''
    # Create datasets
    dataset = ImageFolder(image_path, transform=transform)  # 10,192 + 3,616 = 13,808

    # Shuffle indices to randomize the order of dataset
    torch.manual_seed(42)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    # Define the split ratio for training and testing
    train_size = int(0.8 * len(dataset))

    # Use SubsetRandomSampler for training and testing split
    train_sampler = SubsetRandomSampler(indices[:train_size])
    test_sampler = SubsetRandomSampler(indices[train_size:])

    # Create Dataloaders instances for training and testing
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}\n")

    return train_loader, test_loader


def train(model, learning_rate, epochs, train_loader):
    print("Training the model")
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimization
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)

    # Store training loss
    training_loss = []

    # Train the model
    total_step = len(train_loader)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move images/labels to GPU
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, epochs, i+1, total_step, loss.item()))
                training_loss.append(loss.item())
            
    return model, training_loss

def test(model, test_loader):
    print("\nTesting the model")
    # Store test accuracy
    test_accuracy = []

    """
    Variables for calculating the following metrics
    - accurcay
    - precision
    - recall (true positive rate)
    - specificity (true negative rate)
    """
    TP = 0  # COVID(0), Predicted COVID(0)
    FP = 0  # Normal(1), Predicted COVID(0) 
    TN = 0  # Normal(1), Predicted Normal(1)
    FN = 0 # COVID(0), Predicted Normal(1) 

    with torch.no_grad():
        for images, labels in test_loader:
            # move images/labels to GPU
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  # send batch of images through CNN
            # Tensor.data = tensor
            _, predicted = torch.max(outputs.data, 1)  # extract highest predicted value

            for i in range(len(labels)):
                if labels[i].item() == 1 and predicted[i].item() == 1:
                    TP += 1
                if labels[i].item() == 0 and predicted[i].item() == 1:
                    FP += 1
                if labels[i].item() == 0 and predicted[i].item() == 0:
                    TN += 1
                if labels[i].item() == 1 and predicted[i].item() == 0:
                    FN += 1

            test_accuracy.append(((TP+TN)/(TP+FP+TN+FN))*100)

        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        print(f"Final test accuracy: {((TP+TN)/(TP+FP+TN+FN))*100}%")

    return test_accuracy

def plotLoss(training_loss):
    # Plot training loss
    plt.figure()
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(training_loss)
    plt.show()

def plotAccuracy(test_accuracy):
    # Plot test accuracy
    plt.figure()
    plt.title("Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    plt.plot(test_accuracy)
    plt.show()

if __name__ == "__main__":
    # Import model
    model = ResNet50()

    # Checks if GPU is available, else uses CPU
    device = check_gpu_availability()
    print(device)

    # Move model to GPU, else keep on CPU
    model.to(device)

    # Hyperparameters
    batch_size = 50
    learning_rate = 0.005
    epochs = 5

    # Setup trainload and test_loader
    train_loader, test_loader = preprocessData(batch_size)
    
    # Train the model
    trainedModel, loss = train(model, learning_rate, epochs, train_loader)

    # Plot Loss
    plotLoss(loss)

    # Test the model
    accuracy = test(trainedModel, test_loader)

    # Plot Accuracy
    plotAccuracy(accuracy)
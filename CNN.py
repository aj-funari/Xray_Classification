import torch.nn as nn

# Convolution Neural Network
class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        # input shape: (None,3,28,28)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # input shape: (None,10,12,12)
        self.layer2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # input shape: (None,20,4,4)
        self.fc1 = nn.Linear(4*4*20, 50)

        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0),-1)  # flatten image
        x = self.fc1(x)
        x = self.fc2(x)
        return x
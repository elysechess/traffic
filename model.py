import torch.nn as nn
from torch import flatten


class LeNet(nn.Module):

    # n_channels: 1 for grayscale, 3 for RGB
    # n_pos_out: number of possible output classification decisions
    def __init__(self, n_channels, n_pos_out):

        # Parent constructor (nn.Module)
        super(LeNet, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels = n_channels, out_channels = 6, kernel_size = (5, 5)) # Filter image for feature
        self.relu1 = nn.ReLU() # Detect feature within filtered image
        self.maxpool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)) # Condense image to enhance features

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = (5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2), padding = 1)

        # Dense block
        self.linear1 = nn.Linear(in_features = 400, out_features = 84)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(in_features = 84, out_features = n_pos_out)
        self.logsoftmax = nn.LogSoftmax(dim = 1) # Softmax classifier

    # Override forward method of nn.Module
    def forward(self, x):

        # Pass input x through first convolutional layer
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # Pass output through second convolutional layer
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # Flatten output from second convolutional layer and pass through dense block
        x = flatten(x, 1)
        x = self.linear1(x)
        x = self.relu3(x)

        # Pass output through softmax to obtain classification result
        x = self.linear2(x)
        result = self.logsoftmax(x)
        return result
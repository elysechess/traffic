from PIL import Image
import os
import matplotlib.pyplot as plt
from random import randint
from model import LeNet
from custom_dataset import CustomDataSet
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.nn import NLLLoss
from torch.optim import Adam
import torch
import numpy as np
from time import time
import sys


IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TRAIN_RATIO = 0.8
BATCH_SIZE = 64
INIT_LR = 1e-3
EPOCHS = 10
CATEGORY_KEY = {
    0: "20 km/h",
    1: "30 km/h",
    2: "50 km/h",
    3: "60 km/h", 
    4: "70 km/h",
    5: "80 km/h",
    6: "end 80 km/h",
    7: "100 km/h",
    8: "120 km/h",
    9: "no passing",
    10: "no passing ( > 3.5 mt)",
    11: "right of way",
    12: "priority road",
    13: "yield",
    14: "stop",
    15: "no vehicles",
    16: "no vehicles > 3.5 mt",
    17: "no entry",
    18: "caution",
    19: "watch curve left",
    20: "watch curve right",
    21: "double curve",
    22: "bumpy road",
    23: "slippery road",
    24: "road narrows right",
    25: "road work",
    26: "traffic signals",
    27: "pedestrians",
    28: "children crossing",
    29: "bicycles crossing",
    30: "ice/snow",
    31: "wild animals crossing",
    32: "end of speed/passing limits",
    33: "turn right ahead",
    34: "turn left ahead",
    35: "ahead only",
    36: "straight or right",
    37: "straight or left",
    38: "keep right",
    39: "keep left",
    40: "roundabout",
    41: "end no passing",
    42: "end no passing (> 3.5 mt)"
}

# Accepts path to data, displays subset and returns image arrays and labels
def load_data(path):
    imgs = []
    labels = []

    # Create plot
    fig = plt.figure(figsize = (20, 8))
    plt.title("Examples from Dataset")
    plt.axis("off")
    plt.tight_layout()

    # Iterate through each directory within gtsrb
    for dir in range(NUM_CATEGORIES):
        d_path = os.path.join(path, str(dir))
        print("Processing category " + str(dir) + "...")

        # Iterate through all images within directory
        s_dir = os.listdir(d_path)
        for img_name in s_dir:

            # Get image and resize
            img = Image.open(os.path.join(d_path, str(img_name)))
            r_img = img.resize((IMG_WIDTH, IMG_HEIGHT))

            # Add image and its label to list
            labels.append(dir)
            imgs.append(r_img)

        # Add random image from directory to plot
        ax = fig.add_subplot(5, 9, dir + 1)
        img_name = s_dir[randint(0, len(s_dir) - 1)]
        img = Image.open(os.path.join(d_path, str(img_name)))
        r_img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        ax.imshow(r_img)
        ax.set_title(CATEGORY_KEY[dir], fontsize = 8)
        ax.axis("off")
    print()

    # Display data samples
    plt.show()

    # Return images and corresponding labels
    return (imgs, labels)

# Train the model
def train_model(data):
    
    # Split data into train and test sets
    train_end = int(len(data[0]) * TRAIN_RATIO) # Ending index of training data
    
    # Get (randomized) indices for each set
    indices = np.arange(len(data[0]))
    np.random.shuffle(indices)
    train_i, test_i = indices[:train_end], indices[train_end:]

    # Initialize DataLoaders and input split data
    train_set = CustomDataSet([data[0][i] for i in train_i], [data[1][i] for i in train_i], ToTensor())
    test_set = CustomDataSet([data[0][i] for i in test_i], [data[1][i] for i in test_i], ToTensor())
    train = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
    test = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = False)

    # Initialize model, optimizer and loss function
    model = LeNet(3, NUM_CATEGORIES)
    optim = Adam(params = model.parameters(), lr = INIT_LR)
    loss_func = NLLLoss() # Negative log-likelihood loss

    # Store training history for graphing
    hist = {
        "train_loss": [],
        "train_accuracy": [],
    }

    # Time the training process
    start_time = time()

    # Calculate steps per epoch
    train_steps = len(train.dataset) // BATCH_SIZE

    # Loop over epochs
    print("Beginning training...")
    for epoch in range(EPOCHS):

        # Set model to training mode, initialize total losses and accuracies
        model.train()
        train_loss = 0
        train_correct = 0

        # Loop over training data
        for (x, y) in train:

            # Perform forward pass and calculate training loss
            pred = model(x)
            loss = loss_func(pred, y)

            # Zero gradient, perform backpropagation, update weights
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Update losses and accuracies
            train_loss += loss
            train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Compute, record, and display epoch statistics
        avg_train_loss = train_loss / train_steps # Average train loss
        train_acc = train_correct / len(train.dataset) # Train accuracy
        hist["train_loss"].append(avg_train_loss.detach().numpy())
        hist["train_accuracy"].append(train_acc)
        print("EPOCH: {}/{}".format(epoch + 1, EPOCHS))
        print("Train loss: {:.4f} | Train accuracy: {:.4f}".format(avg_train_loss, train_acc))

    # Stop timing train period
    end_time = time()
    print()
    print("Total training time: {:.2f} seconds".format(end_time - start_time))

    # Evaluate accuracy by disabling gradient calculations
    with torch.no_grad():

        # Set model to evaluation mode
        model.eval()

        # Calculate model accuracy 
        num_correct = 0
        for (x, y) in test:
            pred = model(x)
            num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        print("Model accuracy: {:.4f}%".format((num_correct / len(test.dataset)) * 100))

    # Plot loss and accuracy
    plt.figure()
    plt.plot(hist["train_loss"], label = "Train loss")
    plt.plot(hist["train_accuracy"], label = "Train accuracy")
    plt.legend(loc = "upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Loss + Accuracy")
    plt.title("Training Loss and Accuracy on GTSRB Dataset")
    plt.show()
    return model

def main():

    # Check for proper command line arguments
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        sys.exit("Usage: python traffic.py data_directory [saved_model_name.pt]")

    # Load data and display random subset
    data = load_data(sys.argv[1])

    # Train model and display visual examples 
    model = train_model(data)

    # Save model if desired
    if len(sys.argv) == 3:
        torch.save(model.state_dict(), sys.argv[2])
        print("Model saved to {filename}".format(filename = sys.argv[2]))

if __name__ == "__main__":
    main()
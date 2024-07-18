import cv2
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


IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TRAIN_RATIO = 0.8
BATCH_SIZE = 64
INIT_LR = 1e-3
EPOCHS = 10

# Accepts path to data, returns image arrays and labels
def load_data(path):
    imgs = []
    labels = []

    # Iterate through each directory within gtsrb
    for dir in range(NUM_CATEGORIES):
        d_path = os.path.join(path, str(dir))
        print("Processing category " + str(dir) + "...")

        # Iterate through all images within directory
        for img_name in os.listdir(d_path):

            # Get image and resize
            img = cv2.imread(os.path.join(d_path, str(img_name)))
            r_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_LINEAR)

            # Add image and its label to list
            labels.append(dir)
            imgs.append(r_img)

    # Return images and corresponding labels
    return (imgs, labels)

# Displays random example images from dataset
def disp_some_data(data):
    fig = plt.figure(figsize = (10, 7))
    plt.title("Examples from Dataset")
    plt.axis("off")
    for ex in range(49):
        fig.add_subplot(7, 7, ex + 1)
        rand_ex = data[0][randint(0, len(data[0]) - 1)]
        plt.imshow(rand_ex)
        plt.axis("off")
    plt.show()

# Train the model
def train_model(data, save = False):
    
    # Split data into train, validation, and test sets
    train_end = int(len(data[0]) * TRAIN_RATIO) # Ending index of training data
    val_end = int(train_end + ((len(data[0]) - train_end) / 2)) # Ending index for validation data
    
    # Get (randomized) indices for each set
    indices = np.arange(len(data[0]))
    np.random.shuffle(indices)
    train_i, val_i, test_i = indices[:train_end], indices[train_end:val_end], indices[val_end:]

    # Initialize DataLoaders and input split data
    train_set = CustomDataSet([data[0][i] for i in train_i], [data[1][i] for i in train_i], ToTensor())
    val_set = CustomDataSet([data[0][i] for i in val_i], [data[1][i] for i in val_i], ToTensor())
    test_set = CustomDataSet([data[0][i] for i in test_i], [data[1][i] for i in test_i], ToTensor())
    train = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
    val = DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False)
    test = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = False)

    # Initialize model, optimizer and loss function
    model = LeNet(3, NUM_CATEGORIES)
    optim = Adam(params = model.parameters(), lr = INIT_LR)
    loss_func = NLLLoss() # Negative log-likelihood loss

    # Store training history for graphing and time training
    hist = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }
    start_time = time()

    # Loop over epochs
    for epoch in range(EPOCHS):

        # Set model to training mode, initialize total losses and accuracies
        model.train()
        train_loss = 0
        train_correct = 0
        val_loss = 0
        val_correct = 0

        # Loop over training data
        for (x, y) in train:

            # Perform forward pass and calculate training loss
            pred = model(x)
            loss = loss_func(pred, y)

            # Zero gradient, perform backpropagation, update weights
            optim.zero_grad()
            loss_func.backward()
            optim.step()

            # Update losses and accuracies
            train_loss += loss
            train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Evaluate current state of model by turning off gradient calculations
        with torch.no_grad():

            # Set model to evalulation mode
            model.eval()

            # Loop through validation data
            for (x, y) in val:

                # Make prediction and calculate validation loss
                pred = model(x)
                val_loss += loss_func(pred, y)

                # Calculate correct predictions
                val_correct = (pred.argmax(1) == y).type(torch.float).sum().item()

        # Compute epoch statistics









    # Save model if desired
    if save:
        pass






# Include a function to graph loss, model accuracy over time


# read section 7
# add-on to project: implement one of the more advanced models from section 8
# read section 14


if __name__ == "__main__":
    path = r"C:\Users\elyse\Desktop\traffic\gtsrb"
    data = load_data(path)
    disp_some_data(data)
    train_model(data)

    




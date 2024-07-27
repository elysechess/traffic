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
    print()

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

    # Store training history for graphing
    hist = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    # Time the training process
    start_time = time()

    # Calculate steps per epoch
    train_steps = len(train.dataset) // BATCH_SIZE
    val_steps = len(val.dataset) // BATCH_SIZE

    # Loop over epochs
    print("Beginning training...")
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
            loss.backward()
            optim.step()

            # Update losses and accuracies
            train_loss += loss
            train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Evaluate current state of model by disabling gradient calculations
        with torch.no_grad():

            # Set model to evalulation mode
            model.eval()

            # Loop through validation data
            for (x, y) in val:

                # Make prediction and calculate validation loss
                pred = model(x)
                val_loss += loss_func(pred, y)

                # Calculate correct predictions
                val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Compute, record, and display epoch statistics
        avg_train_loss = train_loss / train_steps # Average train loss
        avg_val_loss = val_loss / val_steps # Average validation loss
        train_acc = train_correct / len(train.dataset) # Train accuracy
        val_acc = val_correct / len(val.dataset) # Validation accuracy
        hist["train_loss"].append(avg_train_loss.detach().numpy())
        hist["train_accuracy"].append(train_acc)
        hist["val_loss"].append(avg_val_loss.detach().numpy())
        hist["val_accuracy"].append(val_acc)
        print("EPOCH: {}/{}".format(epoch + 1, EPOCHS))
        print("Train loss: {:.4f} | Train accuracy: {:.4f}".format(avg_train_loss, train_acc))
        print("Validation loss: {:.4f} | Validation accuracy: {:.4f}".format(avg_val_loss, val_acc))

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
    plt.plot(hist["val_loss"], label = "Validation loss")
    plt.plot(hist["train_accuracy"], label = "Train accuracy")
    plt.plot(hist["val_accuracy"], label = "Validation accuracy")
    plt.legend(loc = "upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Loss + Accuracy")
    plt.title("Training Loss and Accuracy on GTSRB Dataset")
    plt.show()

    # Save model if desired
    if save:
        torch.save(model.state_dict())

# Display model functionality
def test_model(data):

    # Should choose 10 images from test set, predict + display results
    raise NotImplementedError

if __name__ == "__main__":
    path = r"C:\Users\elyse\Desktop\traffic\gtsrb"
    data = load_data(path)
    disp_some_data(data)
    train_model(data)

    




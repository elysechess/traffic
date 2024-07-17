import cv2
import os
import matplotlib.pyplot as plt
import random
from model import LeNet
from custom_dataset import CustomDataSet
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
from torch.optim import Adam


IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43

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
        rand_ex = data[0][random.randint(0, len(data[0]) - 1)]
        plt.imshow(rand_ex)
        plt.axis("off")
    plt.show()

# Train the model
def train_model(data):
    
    # Split data into train and test sets
    dataset = CustomDataSet(data[0], data[1], ToTensor)



    # Create model
    model = LeNet(3, NUM_CATEGORIES)



# Include a function to graph loss, model accuracy over time


# read section 7
# add-on to project: implement one of the more advanced models from section 8
# read section 14


if __name__ == "__main__":
    path = r"C:\Users\elyse\Desktop\traffic\gtsrb"
    data = load_data(path)
    disp_some_data(data)

    # Split data into training and testing sets
    




import sys
from model import LeNet
from traffic import NUM_CATEGORIES, IMG_WIDTH, IMG_HEIGHT, CATEGORY_KEY
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
from PIL import Image
from custom_single_dataset import CustomSingleDataSet
import matplotlib.pyplot as plt

def main():
    
    # Check for proper command line arguments
    if len(sys.argv) != 3:
        sys.exit("Usage: python classify.py saved_model_name.pt image_path")

    # Load model
    model = LeNet(3, NUM_CATEGORIES)
    model.load_state_dict(torch.load(sys.argv[1]))
    model.eval()

    # Load and prepare user image
    img = Image.open(sys.argv[2]).convert("RGB")
    r_img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    set = CustomSingleDataSet(img = r_img, transform = ToTensor())
    load = DataLoader(set, batch_size = 1, shuffle = False)

    # Classify image
    with torch.no_grad():
        for image in load:
            output = model(image)
            _, predicted = torch.max(output, 1)
            print("Prediction:", CATEGORY_KEY[predicted.item()])

    # Display result
    plt.figure()
    plt.imshow(img)
    plt.title(CATEGORY_KEY[predicted.item()])
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
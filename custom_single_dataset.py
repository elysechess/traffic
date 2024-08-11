from torch.utils.data import Dataset


# Overload Dataset class to allow input of custom dataset (single image)
class CustomSingleDataSet(Dataset):

    # Initialize
    def __init__(self, img, transform = None):
        self.img = img # Single image
        self.transform = transform # Any transform for image input

    # Return 1 to indicate single image
    def __len__(self):
        return 1
    
    # Return the image
    def __getitem__(self, i):
        if self.transform:
            self.img = self.transform(self.img)
        return self.img
from torch.utils.data import Dataset


# Overload Dataset class to allow input of custom dataset
class CustomDataSet(Dataset):

    # Initialize
    def __init__(self, imgs, labels, transform = None, target_transform = None):
        self.imgs = imgs # Images in dataset
        self.labels = labels # Corresponding labels
        self.transform = transform # Any transform for image inputs
        self.target_transform = target_transform # Any transform for label inputs

    # Return number of samples in dataset
    def __len__(self):
        return len(self.labels)
    
    # Return sample from dataset at given index
    def __getitem__(self, i):
        img = self.imgs[i]
        label = self.labels[i]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label
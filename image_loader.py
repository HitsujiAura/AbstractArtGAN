import os
from torch.utils.data import Dataset, DataLoader


class AbstractArtDataset(Dataset):
    def __init__(self, image_path, transforms):
        self.transforms = transforms
        self.image_path = image_path
        self.image_names = [name for name in os.listdir(self.image_path) if os.path.isfile(os.path.join(self.image_path, name))]
    def __len__ (self):
        return len(self.image_names)
    def __getitem__(self, idx):
        image_location = os.path.join(self.image_path, self.image_names[idx])
        image = Image.open(image_location).convert('RGB')
        image = self.transforms(image)
        return image

import h5py
from pathlib import Path
from torch.utils.data import Dataset


class PCamDataset(Dataset):

    def __init__(self, image_path, label_path, transform=None):
        self.image_path = Path(image_path)
        self.label_path = Path(label_path)

        self.transform = transform

        self.image_file = h5py.File(self.image_path, "r")
        self.label_file = h5py.File(self.label_path, "r")

        self.images = self.image_file["x"]
        self.labels = self.label_file["y"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index].item()

        if self.transform is not None:
            image = self.transform(image)

        return image, label
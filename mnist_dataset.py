from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch

class MNISTDataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.dataset = datasets.MNIST(root=path, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        return img, target

class MNISTWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset_dir, train=True):
        super().__init__()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Mean and std computed from MNIST dataset
        ])
        self.dataset = MNISTDataset(path=dataset_dir, train=train, transform=transform)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch

class CIFAR10Dataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.dataset = datasets.CIFAR10(root=path, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        return img, target

class CIFAR10Wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, train=True):
        super().__init__()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Mean and std computed from CIFAR10 dataset
        ])
        self.dataset = CIFAR10Dataset(path=dataset_dir, train=train, transform=transform)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

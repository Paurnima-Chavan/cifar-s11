from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch.utils.data


class CIFAR10Dataset(datasets.CIFAR10):

    def __init__(self, root="~/data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


def load_cifar10_data(batch_size):
    """
   This function is designed to retrieve the CIFAR10 dataset from the PyTorch library, apply several transformations
   to it, and ultimately provide data loaders for both the training and testing sets.
    :param batch_size: Batch size refers to the number of samples or data points that are processed simultaneously
                       or at a given time during the training or inference process.
    :return: Data loader object for both the training and testing sets    """

    means = [0.4914, 0.4822, 0.4465]
    stds = [0.2470, 0.2435, 0.2616]

    train_transforms = A.Compose(
        [
            A.Normalize(mean=means, std=stds, always_apply=True),
            A.PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_CONSTANT, value=4, always_apply=True),
            A.RandomCrop(height=32, width=32, always_apply=True),
            A.HorizontalFlip(),
            A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8,
                            fill_value=means),
            ToTensorV2(),
        ]
    )

    test_transforms = A.Compose(
        [
            A.Normalize(mean=means, std=stds, always_apply=True),
            ToTensorV2(),
        ]
    )

    train_dataset = CIFAR10Dataset(root='./data', train=True, download=True, transform=train_transforms)
    test_dataset = CIFAR10Dataset(root='./data', train=False, download=True, transform=test_transforms)

    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)

    return train_loader, test_loader

import torch

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ratio of data to use for validation
valid_split = 0.2
# batch size
batch_size = 64
# path to the data root directory
root_dir = 'Dataset/'

# define the training transforms and augmentations
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# initial entire and test datasets
dataset = datasets.ImageFolder(root_dir, transform=train_transform)
dataset_test = datasets.ImageFolder(root_dir, transform=valid_transform)

dataset_size = len(dataset)

valid_size = int(valid_split*dataset_size)

# training and validation sets
indices = torch.randperm(len(dataset)).tolist()
dataset_train = Subset(dataset, indices[:-valid_size])
dataset_valid = Subset(dataset_test, indices[-valid_size:])

# training and validation data loaders
train_loader = DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, num_workers=3
)
valid_loader = DataLoader(
    dataset_valid, batch_size=batch_size, shuffle=False, num_workers=3
)
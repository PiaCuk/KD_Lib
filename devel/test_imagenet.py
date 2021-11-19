import os
import torch
from torchvision import datasets, transforms

data_dir = "/home/pia/Documents/ImageNet"
batch_size = 2
workers = 1

# Data loading code
traindir = os.path.join(data_dir, 'train')
valdir = os.path.join(data_dir, 'val')
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
train_sampler = None

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]),
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=(train_sampler is None),
    num_workers=workers,
    pin_memory=True,
    sampler=train_sampler,
)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers,
    pin_memory=True,
)

print(next(iter(val_loader)))

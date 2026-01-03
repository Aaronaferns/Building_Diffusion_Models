import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def make_cifar10_train_loader(
    data_root="./data",
    batch_size=128,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
):
    tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),                          # [0,1]
        transforms.Normalize((0.5, 0.5, 0.5),           # -> [-1,1]
                             (0.5, 0.5, 0.5)),
    ])

    ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )
    return loader

def make_cifar10_eval_loader(data_root="./data", batch_size=128, num_workers=4):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
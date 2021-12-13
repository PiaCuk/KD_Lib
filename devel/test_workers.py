import os
import time
import torch
from torchvision import datasets, transforms

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

pin_memory = True
print('pin_memory is', pin_memory)

train_data = datasets.FashionMNIST(
    "data/FashionMNIST",
    train=True,
    download=False,
    transform=transforms.Compose([
        # transforms.Grayscale(num_output_channels=3), transforms.RandAugment(),
        transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))
        ]),
)

for num_workers in range(10, 17, 1):
    for b in [512, 1024]:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=b,
                                                   num_workers=num_workers, pin_memory=pin_memory,
                                                   persistent_workers=True)
        start = time.time()
        for epoch in range(1, 5):
            for i, data in enumerate(train_loader):
                pass
        end = time.time()
        print("Finish with:{} second, num_workers={}, batch_size={}".format(
            end - start, num_workers, b))

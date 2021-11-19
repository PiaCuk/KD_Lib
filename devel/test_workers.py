import time
import torch
from torchvision import datasets, transforms


pin_memory = True
print('pin_memory is', pin_memory)

train_data = datasets.FashionMNIST(
    "FashionMNIST",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    ),
)

for num_workers in range(8, 17, 1):
    for b in [512, 1024]:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=b,
                                                   num_workers=num_workers, pin_memory=pin_memory)
        start = time.time()
        for epoch in range(1, 5):
            for i, data in enumerate(train_loader):
                pass
        end = time.time()
        print("Finish with:{} second, num_workers={}, batch_size={}".format(
            end - start, num_workers, b))

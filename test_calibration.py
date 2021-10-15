import torch
from torch import nn
from torch._C import device
from torchvision import datasets, transforms
from KD_Lib.models.resnet import ResNet18
from temperature_scaling import ModelWithTemperature


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "FashionMNIST",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
            ),
        ),
        batch_size=1024,
        shuffle=True,
        pin_memory=True,
        num_workers=16,
        worker_init_fn=None,
        generator=None,
    )

state_dict = torch.load("/data1/9cuk/kd_lib/session7/tfkd002/student.pt")
model = ResNet18([4, 4, 4, 4, 4], 1, 10).to(device)
model.load_state_dict(state_dict)
model.eval()

scaled_model = ModelWithTemperature(model)
scaled_model.set_temperature(test_loader)
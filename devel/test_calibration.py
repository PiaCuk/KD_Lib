import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from ..KD_Lib.models.resnet import ResNet18
from ..temperature_scaling import ModelWithTemperature, _ECELoss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1024
num_workers = 4

fmnist_test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
        "FashionMNIST",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
        ),
    ),
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=num_workers,
    worker_init_fn=None,
    generator=None,
)
"""
cifar10_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "CIFAR10",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=None,
        generator=None,
    )

state_dict = torch.load("/data1/9cuk/kd_lib/session7/tfkd002/student.pt")
model = ResNet18([4, 4, 4, 4, 4], 1, 10).to(device)
model.load_state_dict(state_dict)
model.eval()

scaled_model = ModelWithTemperature(model)
scaled_model.set_temperature(fmnist_test_loader)
"""
# Calculate temperature for virtual model


def temperature_scale(logits, temperature):
    # Expand temperature to match the size of logits
    temperature = temperature.unsqueeze(
        1).expand(logits.size(0), logits.size(1))
    return logits / temperature


temperature = nn.Parameter(torch.ones(1, requires_grad=True) * 10).to(device)
num_classes = 10
correct_prob = 0.99

nll_criterion = nn.CrossEntropyLoss().to(device)
ece_criterion = _ECELoss().to(device)

# First: collect all the logits and labels for the validation set
logits_list = []
labels_list = []
with torch.no_grad():
    for _, label in fmnist_test_loader:
        logits = torch.ones(label.shape[0], num_classes).to(device)
        logits = logits * (1 - correct_prob) / (num_classes - 1)
        for i in range(label.shape[0]):
            logits[i, label[i]] = correct_prob
        logits_list.append(logits)
        labels_list.append(label)

    logits = torch.cat(logits_list).to(device)
    labels = torch.cat(labels_list).to(device)

# Calculate NLL and ECE before temperature scaling
label_ece = ece_criterion(F.one_hot(torch.LongTensor(
    labels), num_classes).type(torch.Tensor), labels).item()
print('Hard labels ECE: %.3f' % (label_ece))

before_temperature_nll = nll_criterion(logits, labels).item()
before_temperature_ece = ece_criterion(logits, labels).item()
print('Before temperature: %.3f' % temperature.item())
print('Before temperature - NLL: %.3f, ECE: %.3f' %
      (before_temperature_nll, before_temperature_ece))

# Next: optimize the temperature w.r.t. NLL
optimizer = optim.LBFGS([temperature], lr=0.01,
                        max_iter=100, line_search_fn='strong_wolfe')


def eval():
    optimizer.zero_grad()
    loss = nll_criterion(temperature_scale(logits, temperature), labels)
    loss.backward()
    return loss


optimizer.step(eval)
temperature = temperature.clamp(min=1.0)

# Calculate NLL and ECE after temperature scaling
after_temperature_nll = nll_criterion(
    temperature_scale(logits, temperature), labels).item()
after_temperature_ece = ece_criterion(
    temperature_scale(logits, temperature), labels).item()
print('Optimal temperature: %.3f' % temperature.item())
print('After temperature - NLL: %.3f, ECE: %.3f' %
      (after_temperature_nll, after_temperature_ece))

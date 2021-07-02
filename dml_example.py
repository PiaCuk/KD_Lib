import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import datasets, transforms

from KD_Lib.KD import DML, DMLEnsemble
from KD_Lib.models import Shallow, ResNet18


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

save_path = "/data1/9cuk/kd_lib/dml1"
epochs = 100
lr = 0.001
batch_size = 512
num_students = 2


class CustomKLDivLoss(nn.Module):
    def __init__(self, reduction='batchmean', log_target=False) -> None:
        super(CustomKLDivLoss, self).__init__()
        self.reduction = reduction
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.kl_div(torch.log_softmax(input, dim=-1), torch.softmax(target, dim=-1),
                        reduction=self.reduction, log_target=self.log_target)


train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
        "FashionMNIST",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
        "FashionMNIST",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

# Set device to be trained on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define models
resnet_params = ([4, 4, 4, 4, 4], 1, 10)
student_cohort = [ResNet18(*resnet_params) for i in range(num_students)]

student_optimizers = [torch.optim.Adam(student_cohort[i].parameters(), lr) for i in range(num_students)]

# Define DML
dml = DML(student_cohort, train_loader, test_loader,
          student_optimizers, loss_fn=CustomKLDivLoss(), log=True, logdir=save_path, device=device)
dml.get_parameters()

# Run DML
dml.train_students(epochs=epochs, save_model=True, save_model_path=save_path)

# Evaluate students
dml.evaluate()

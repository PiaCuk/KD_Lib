import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import datasets, transforms

from KD_Lib.KD import DML
from KD_Lib.models import Shallow, ResNet18


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
    batch_size=512,
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
    batch_size=512,
    shuffle=True,
)

# Set device to be trained on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define models
resnet_params = ([4, 4, 4, 4, 4], 1, 10)
student1 = ResNet18(*resnet_params)
student2 = ResNet18(*resnet_params)
student_cohort = [student1, student2]

student_optimizer1 = torch.optim.Adam(student1.parameters(), 0.001)
student_optimizer2 = torch.optim.Adam(student2.parameters(), 0.001)
student_optimizers = [student_optimizer1, student_optimizer2]

# Define DML
dml = DML(student_cohort, train_loader, test_loader,
          student_optimizers, loss_fn=CustomKLDivLoss(), log=True, device=device)
dml.get_parameters()

# Run DML
dml.train_students(epochs=10, save_model=False)

# Evaluate students
dml.evaluate()

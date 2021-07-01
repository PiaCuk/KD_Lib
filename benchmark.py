import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import datasets, transforms

from KD_Lib.KD import VanillaKD, DML
from KD_Lib.models import Shallow, ResNet18


class CustomKLDivLoss(nn.Module):
    def __init__(self, reduction='batchmean', log_target=False) -> None:
        super(CustomKLDivLoss, self).__init__()
        self.reduction = reduction
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.kl_div(torch.log_softmax(input, dim=-1), torch.softmax(target, dim=-1),
                        reduction=self.reduction, log_target=self.log_target)


def create_distiller(algo, lr, train_loader, test_loader, device, save_path, loss_fn=CustomKLDivLoss()):
    resnet_params = ([4, 4, 4, 4, 4], 1, 10)
    if algo is "dml":
        # Define models
        student1 = ResNet18(*resnet_params)
        student2 = ResNet18(*resnet_params)
        student_cohort = [student1, student2]

        student_optimizer1 = torch.optim.Adam(student1.parameters(), lr)
        student_optimizer2 = torch.optim.Adam(student2.parameters(), lr)
        student_optimizers = [student_optimizer1, student_optimizer2]
        # Define DML
        distiller = DML(student_cohort, train_loader, test_loader, student_optimizers, loss_fn=loss_fn, log=True, logdir=save_path, device=device)
        
    else:
        teacher = ResNet18(*resnet_params)
        student = ResNet18(*resnet_params)

        teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr)
        student_optimizer = torch.optim.Adam(student.parameters(), lr)
        # TODO logging
        distiller = VanillaKD(teacher, student, train_loader, test_loader, teacher_optimizer, student_optimizer, loss_fn=loss_fn, device=device)
    return distiller


def main(algo, runs, epochs, batch_size, lr, save_path):
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
    
    for i in range(runs):
        print(f"Starting run {i}")
        run_path = os.path.join(save_path, algo + str(i).zfill(3))
        distiller = create_distiller(algo, lr, train_loader, test_loader, device, save_path=run_path)
        
        # Run DML
        distiller.train_students(epochs=epochs, save_model=True, save_model_path=run_path)
        # Evaluate students
        distiller.evaluate()


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    main("dml", 10, 100, 512, 0.001, "/data1/9cuk/kd_lib/")
    main("vanilla", 10, 100, 512, 0.001, "/data1/9cuk/kd_lib/")
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


def create_distiller(algo, train_loader, test_loader, device, save_path, loss_fn=CustomKLDivLoss(), use_adam=True):
    def _create_optim(params, adam=False):
        # These are the optimizers used by Zhang et al.
        if adam:
            return torch.optim.Adam(params, lr=0.01, betas=(0.9, 0.999))
        else:
            # Zhang et al. use no weight decay and nesterov=True
            return torch.optim.SGD(params, 0.1, momentum=0.9, weight_decay=0.0001)

    resnet_params = ([4, 4, 4, 4, 4], 1, 10)
    if algo is "dml":
        # Define models
        student1 = ResNet18(*resnet_params)
        student2 = ResNet18(*resnet_params)
        student_cohort = [student1, student2]

        student_optimizer1 = _create_optim(
            student1.parameters(), adam=use_adam)
        student_optimizer2 = _create_optim(
            student2.parameters(), adam=use_adam)
        student_optimizers = [student_optimizer1, student_optimizer2]
        # Define DML with logging to Tensorboard
        distiller = DML(student_cohort, train_loader, test_loader, student_optimizers,
                        loss_fn=loss_fn, log=True, logdir=save_path, device=device, use_scheduler=True)

    else:
        teacher = ResNet18(*resnet_params)
        student = ResNet18(*resnet_params)

        teacher_optimizer = _create_optim(teacher.parameters(), adam=use_adam)
        student_optimizer = _create_optim(student.parameters(), adam=use_adam)
        # Define KD with logging to Tensorboard
        distiller = VanillaKD(teacher, student, train_loader, test_loader, teacher_optimizer,
                              student_optimizer, loss_fn=loss_fn, log=True, logdir=save_path, device=device)
    return distiller


def main(algo, runs, epochs, batch_size, save_path, use_adam=True):
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
        pin_memory=True,
        num_workers=14,
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
        pin_memory=True,
        num_workers=14,
    )

    # Set device to be trained on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(runs):
        print(f"Starting run {i}")
        run_path = os.path.join(save_path, algo + str(i).zfill(3))
        distiller = create_distiller(
            algo, train_loader, test_loader, device, save_path=run_path, use_adam=use_adam)

        if algo is "dml":
            # Run DML
            distiller.train_students(
                epochs=epochs, save_model=True, save_model_path=run_path, plot_losses=False)
            # Evaluate students
            # Not needed here, as we log it at the end of each epoch
            # distiller.evaluate()
        else:
            distiller.train_teacher(
                epochs=epochs, plot_losses=False, save_model=False)
            distiller.train_student(
                epochs=epochs, plot_losses=False, save_model=True, save_model_pth=run_path)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # main("dml", 5, 100, 2048, "/data1/9cuk/kd_lib/session1", use_adam=True)
    main("vanillla", 5, 100, 2048, "/data1/9cuk/kd_lib/session1", use_adam=True)
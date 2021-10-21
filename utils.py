import random
import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import datasets, transforms

from KD_Lib.KD import VanillaKD, DML, VirtualTeacher
from KD_Lib.models import Shallow, ResNet18, ResNet50


class CustomKLDivLoss(torch.nn.Module):
    def __init__(self, reduction='batchmean', log_target=False) -> None:
        super(CustomKLDivLoss, self).__init__()
        self.reduction = reduction
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.kl_div(torch.log_softmax(input, dim=-1), torch.softmax(target, dim=-1), reduction=self.reduction, log_target=self.log_target)


class SoftKLDivLoss(torch.nn.Module):
    def __init__(self, temp=20.0, reduction='batchmean', log_target=False) -> None:
        super(SoftKLDivLoss, self).__init__()
        self.temp = temp
        self.reduction = reduction
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        soft_input = torch.log_softmax(input / self.temp, dim=-1)
        soft_target = torch.softmax(target / self.temp, dim=-1)
        # Multiply with squared temp so that KLD loss keeps proportion to CE loss
        return (self.temp ** 2) * F.kl_div(soft_input, soft_target, reduction=self.reduction, log_target=self.log_target)


def set_seed(seed=42, cuda_deterministic=False) -> torch.Generator:
    # See https://stackoverflow.com/a/64584503/8697610
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    # Generator used to control sampling of dataset
    # See https://pytorch.org/docs/stable/data.html#data-loading-randomness
    g = torch.Generator()
    g.manual_seed(torch.initial_seed())
    return g


def seed_worker(worker_id):
    # See https://pytorch.org/docs/stable/notes/randomness.html
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(batch_size, train, generator=None):
    return torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "FashionMNIST",
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16,
        worker_init_fn=seed_worker if generator is not None else None,
        generator=generator,
    )


def _create_optim(params, lr, adam=False):
    # These are the optimizers used by Zhang et al.
    if adam:
        return torch.optim.Adam(params, lr, betas=(0.9, 0.999))
    else:
        # Zhang et al. use no weight decay and nesterov=True
        return torch.optim.SGD(params, lr, momentum=0.9, weight_decay=0.0001)


def _create_scheduler(optim, lr, epochs, epoch_len):
    return torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=lr, epochs=epochs, steps_per_epoch=epoch_len, pct_start=0.1)


def create_distiller(algo, train_loader, test_loader, device, save_path, loss_fn=CustomKLDivLoss(), lr=0.01, distil_weight=0.5, num_students=2, use_adam=True, use_scheduler=False):
    """
    Create distillers for benchmarking.

    :param algo (str): Name of the training algorithm to use. Either "dml", "dml_e", else VanillaKD
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param test_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param device (str): Device used for training
    :param save_path (str): Directory for storing logs and saving models
    :param loss_fn (torch.nn.Module): Loss Function used for distillation. Not used for VanillaKD (BaseClass), as it is implemented internally
    :param num_students (int): Number of students in cohort. Used for DML
    :param use_adam (bool): True to use Adam optim
    """
    resnet_params = ([4, 4, 4, 4, 4], 1, 10)
    if algo == "dml" or algo == "dml_e":
        # Define models
        student_cohort = [ResNet18(*resnet_params)
                          for i in range(num_students)]
        # student_cohort = [
        #     ResNet50(*resnet_params), ResNet18(*resnet_params), ResNet18(*resnet_params)]

        student_optimizers = [_create_optim(
            student_cohort[i].parameters(), lr, adam=use_adam) for i in range(num_students)]
        # Define DML with logging to Tensorboard
        distiller = DML(student_cohort, train_loader, test_loader, student_optimizers, loss_fn=loss_fn, distil_weight=distil_weight,
                        log=True, logdir=save_path, device=device, use_scheduler=True, use_ensemble=True if algo == "dml_e" else False)
    elif algo == "tfkd":
        student = ResNet18(*resnet_params)
        student_optimizer = _create_optim(
            student.parameters(), lr, adam=use_adam)
        # Define TfKD with logging to Tensorboard
        # Note that we need to use Pytorch's KLD here, due to the implementation of TfKD in KD-Lib
        distiller = VirtualTeacher(student, train_loader, test_loader, student_optimizer, loss_fn=torch.nn.KLDivLoss(
            reduction='batchmean'), distil_weight=distil_weight, log=True, logdir=save_path, device=device)
    else:
        teacher = ResNet50(*resnet_params)
        student = ResNet18(*resnet_params)

        teacher_optimizer = _create_optim(
            teacher.parameters(), lr, adam=use_adam)
        student_optimizer = _create_optim(
            student.parameters(), lr, adam=use_adam)
        # Define KD with logging to Tensorboard
        distiller = VanillaKD(teacher, student, train_loader, test_loader, teacher_optimizer, student_optimizer,
                              loss_fn=loss_fn, distil_weight=distil_weight, log=True, logdir=save_path, device=device)
    return distiller

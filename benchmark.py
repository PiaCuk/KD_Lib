import os
import random
import numpy as np

import torch
from torchvision import datasets, transforms

from utils import CustomKLDivLoss, SoftKLDivLoss, set_seed, create_dataloader, create_distiller
from temperature_scaling import ModelWithTemperature


# TODO implement dynamic temperature in TfKD


def main(algo, runs, epochs, batch_size, save_path, loss_fn=CustomKLDivLoss(), num_students=2, use_adam=True, seed=None):
    """
    Main function to call for benchmarking.

    :param algo (str): Name of the training algorithm to use. Either "dml", "dml_e", else VanillaKD
    :param runs (int): Number of runs for each algorithm
    :param epochs (int): Number of epochs to train per run
    :param batch_size (int): Batch size for training
    :param save_path (str): Directory for storing logs and saving models
    :param loss_fn (torch.nn.Module): Loss Function used for distillation. Not used for VanillaKD (BaseClass), as it is implemented internally
    :param num_students (int): Number of students in cohort. Used for DML
    :param use_adam (bool): True to use Adam optim
    """
    if seed is not None:
        g = set_seed(seed)

    train_loader = create_dataloader(
        batch_size, train=True, generator=g if seed is not None else None)

    test_loader = create_dataloader(
        batch_size, train=False, generator=g if seed is not None else None)

    # Set device to be trained on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(runs):
        print(f"Starting run {i}")
        run_path = os.path.join(save_path, algo + str(i).zfill(3))
        distiller = create_distiller(
            algo, train_loader, test_loader, device, save_path=run_path, loss_fn=loss_fn, num_students=num_students, use_adam=use_adam)

        if algo == "dml" or algo == "dml_e":
            print("Using " + algo)
            # Run DML
            distiller.train_students(
                epochs=epochs, plot_losses=False, save_model=True, save_model_path=run_path)
        elif algo == "tfkd":
            distiller.train_student(
                epochs=epochs, plot_losses=False, save_model=True, save_model_path=run_path)
        else:
            distiller.train_teacher(
                epochs=epochs, plot_losses=False, save_model=False)
            
            scaled_model = ModelWithTemperature(distiller.teacher_model)
            scaled_model.set_temperature(test_loader)
            distiller.temp = scaled_model.temperature.item()
            print("Train student with temperature " + str(distiller.temp))
            
            distiller.train_student(
                epochs=epochs, plot_losses=False, save_model=True, save_model_path=run_path)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    """
    main("dml", 5, 100, 1024, "/data1/9cuk/kd_lib/session6",
         loss_fn=SoftKLDivLoss(), num_students=3)
    main("dml_e", 5, 100, 1024, "/data1/9cuk/kd_lib/session6",
         loss_fn=SoftKLDivLoss(), num_students=3)
    main("vanilla", 5, 100, 1024, "/data1/9cuk/kd_lib/session3_3")
    main("tfkd", 5, 100, 1024, "/data1/9cuk/kd_lib/session7")
    """
    # main("dml", 5, 100, 1024, "/data1/9cuk/kd_lib/calibration1",
    #     loss_fn=CustomKLDivLoss(), num_students=3, seed=42)
    # main("dml_e", 5, 100, 1024, "/data1/9cuk/kd_lib/calibration1",
    #     loss_fn=CustomKLDivLoss(), num_students=3, seed=42)
    # main("vanilla", 5, 100, 1024, "/data1/9cuk/kd_lib/calibration1", seed=42)
    main("tfkd", 5, 100, 1024, "/data1/9cuk/kd_lib/calibration1", seed=42)

import os

import torch

from utils import CustomKLDivLoss, SoftKLDivLoss, set_seed, create_dataloader, create_distiller
from temperature_scaling import ModelWithTemperature


def main(
    algo,
    runs,
    epochs,
    batch_size,
    save_path,
    loss_fn=CustomKLDivLoss(),
    lr=0.005,
    distil_weight=0.5,
    temperature=10,
    num_students=2,
    use_pretrained=False,
    use_scheduler=False,
    seed=None,
):
    """
    Universal main function for my 

    :param algo (str): Name of the training algorithm to use. Either "dml", "dml_e", else VanillaKD
    :param runs (int): Number of runs for each algorithm
    :param epochs (int): Number of epochs to train per run
    :param batch_size (int): Batch size for training
    :param save_path (str): Directory for storing logs and saving models
    :param loss_fn (torch.nn.Module): Loss Function used for distillation. Not used for VanillaKD (BaseClass), as it is implemented internally
    :param lr (float): Learning rate
    :param distil_weight (float): Between 0 and 1
    :param num_students (int): Number of students in cohort. Used for DML
    :use_pretrained (bool): Use pretrained teacher for VanillaKD
    :param use_scheduler (bool): True to decrease learning rate during training
    :param seed:    
    """
    if seed is not None:
        # Set seed for all libraries and return torch.Generator
        g = set_seed(seed)

    # Create DataLoaders
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
            algo, train_loader, test_loader, device, save_path=run_path,
            loss_fn=loss_fn, lr=lr, distil_weight=distil_weight, temperature=temperature, num_students=num_students)
        
        # epochs, plot_losses, save_model, save_model_path, use_scheduler
        param_list = [epochs, False, True, run_path, use_scheduler]

        if algo == "dml" or algo == "dml_e":
            # Run DML or DML_e
            distiller.train_students(*param_list)
        elif algo == "tfkd":
            distiller.train_student(*param_list)
        else:
            if use_pretrained:
                # Use pre-trained teacher to save computation
                state_dict = torch.load(
                    "/data1/9cuk/kd_lib/saved_models/vanilla000/teacher.pt")
                distiller.teacher_model.load_state_dict(state_dict)

                # Optimal temperature found with LBFGS
                # scaled_model = ModelWithTemperature(distiller.teacher_model)
                # scaled_model.set_temperature(test_loader)
                # distiller.temp = scaled_model.temperature.item()
                distiller.temp = 1.243
            else:
                # Train teacher from scratch and save the model
                distiller.train_teacher(*param_list)

            # Train student from scratch
            distiller.train_student(*param_list)

import os
import glob

import torch

from utils import CustomKLDivLoss, SoftKLDivLoss, set_seed, create_dataloader, create_weighted_dataloader, create_distiller


def eval(
    algo,
    load_dir,
    batch_size,
    loss_fn=CustomKLDivLoss(),
    lr=0.005,
    distil_weight=0.5,
    temperature=10.0,
    use_weighted_dl=None,
    seed=None,
):
    """
    Evaluation function for trained and saved models

    :param algo (str): Name of the training algorithm to use. Either "dml", "dml_e", else VanillaKD
    :param load_dir (str):
    :param batch_size (int): Batch size for training
    :param loss_fn (torch.nn.Module): Loss Function used for distillation. Not used for VanillaKD (BaseClass), as it is implemented internally
    :param lr (float): Learning rate
    :param distil_weight (float): Between 0 and 1
    :param temperature (float): temperature parameter for soft targets
    :param use_weighted_dl (bool): True to use weighted DataLoader with oversampling
    :param seed:    
    """
    # Set seed for all libraries and return torch.Generator
    g = set_seed(seed) if seed is not None else None

    # Create DataLoaders
    if isinstance(use_weighted_dl, float):
        print(f"Using weighted dataloader with p(y=0) = {use_weighted_dl}.")
        train_loader = create_weighted_dataloader(
            batch_size, train=True, generator=g, class_weight=use_weighted_dl, workers=8)
        test_loader = create_weighted_dataloader(
            batch_size, train=False, generator=g, class_weight=use_weighted_dl, workers=8)
    else:
        train_loader = create_dataloader(batch_size, train=True, generator=g, workers=8)
        test_loader = create_dataloader(batch_size, train=False, generator=g, workers=8)

    # Set device to be trained on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    distiller = create_distiller(
        algo, train_loader, test_loader, device, save_path=load_dir,
        loss_fn=loss_fn, lr=lr, distil_weight=distil_weight, temperature=temperature, num_students=1)

    if algo == "vanilla":
        eval_teacher = True
        if eval_teacher:
            state_dict = torch.load(os.path.join(load_dir, "teacher.pt"), map_location=device)
            distiller.teacher_model.load_state_dict(state_dict)
        else:
            state_dict = torch.load(os.path.join(load_dir, "student.pt"), map_location=device)
            distiller.student_model.load_state_dict(state_dict)

        if eval_teacher:
            print("Evaluating teacher... \n")
            distiller.evaluate(teacher=True)
        else:
            print("Evaluating student... \n")
            distiller.evaluate(teacher=False)
    
    elif algo == "tfkd":
        state_dict = torch.load(os.path.join(load_dir, "student.pt"), map_location=device)
        distiller.student_model.load_state_dict(state_dict)
        distiller.evaluate()
    
    else:
        model_pt = glob.glob(os.path.join(load_dir, "student*.pt"))[0]
        state_dict = torch.load(model_pt, map_location=device)
        distiller.student_cohort[0].load_state_dict(state_dict)
        distiller.evaluate()


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    for algo in ["vanilla"]: #"vanilla", "tfkd"
        print(f"---------Evaluating {algo}---------")
        for w in [None, 0.9]:
            for i in range(5):
                eval(
                    algo,
                    f"./cvpc8/replicate1/oversample0.9/{algo}00{i}",
                    1,
                    loss_fn=CustomKLDivLoss(apply_softmax=True),
                    lr=0.005,
                    distil_weight=0.5,
                    temperature=10,
                    use_weighted_dl=w,
                    seed=42,
                )

import os

import torch

from utils import CustomKLDivLoss, set_seed, create_dataloader, create_distiller


def main(runs, epochs, batch_size, save_path, distil_weight=0.5, use_pretrained=False, seed=None):
    if seed is not None:
        g = set_seed(seed)

    train_loader = create_dataloader(
        batch_size, train=True, generator=g if seed is not None else None)

    test_loader = create_dataloader(
        batch_size, train=False, generator=g if seed is not None else None)

    # Set device to be trained on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    algo = "CE_only"

    for i in range(runs):
        print(f"Starting run {i}")
        run_path = os.path.join(save_path, algo + str(i).zfill(3))
        distiller = create_distiller(
            algo, train_loader, test_loader, device, save_path=run_path, loss_fn=CustomKLDivLoss(), distil_weight=distil_weight)

        if use_pretrained:
            state_dict = torch.load("/data1/9cuk/kd_lib/saved_models/vanilla000/teacher.pt")
            distiller.teacher_model.load_state_dict(state_dict)
            # Optimal temperature found with LBFGS
            distiller.temp = 1.243
        else:
            distiller.train_teacher(
                epochs=epochs, plot_losses=False, save_model=True, save_model_path=run_path)
        
        distiller.train_student(
            epochs=epochs, plot_losses=False, save_model=True, save_model_path=run_path)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    main(5, 100, 1024, "/data1/9cuk/kd_lib/calibration2",
         distil_weight=0, use_pretrained=True, seed=42)

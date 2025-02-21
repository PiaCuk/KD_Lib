import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical

import matplotlib.pyplot as plt
from copy import deepcopy
import os
from tqdm import tqdm


class DMLEnsemble:
    """
    Implementation of "Deep Mutual Learning with ensemble teacher" https://arxiv.org/abs/1706.00384

    :param student_cohort (list/tuple): Collection of student models
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param student_optimizers (list/tuple): Collection of Pytorch optimizers for training students
    :param loss_fn (torch.nn.Module): Loss Function used for distillation
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    :param use_scheduler(bool): True to decrease learning rate during training
    """

    def __init__(
        self,
        student_cohort,
        train_loader,
        val_loader,
        student_optimizers,
        loss_fn=nn.MSELoss(),
        device="cpu",
        log=False,
        logdir="./Experiments",
        use_scheduler=True
    ):

        self.student_cohort = student_cohort
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.student_optimizers = student_optimizers
        self.loss_fn = loss_fn
        self.log = log
        self.logdir = logdir
        self.student_schedulers = []

        if self.log:
            self.writer = SummaryWriter(logdir)
        
        if use_scheduler:
            # Drop lr by 0.1 every 60 epochs (Zhang et al.)
            for i in range(len(self.student_cohort)):
                self.student_schedulers.append(torch.optim.lr_scheduler.StepLR(
                    self.student_optimizers[i], step_size=20, gamma=0.1))

        if device.type == "cpu":
            self.device = torch.device("cpu")
            print("Device is set to CPU.")
        elif device.type == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Device is set to CUDA.")
        else:
            print(
                "Either an invalid device or CUDA is not available. Defaulting to CPU."
            )
            self.device = torch.device("cpu")
        
        for student in self.student_cohort:
            student.to(self.device)
    
    def ensemble_target(self, logits_list, j):
        # Calculate ensemble target given a list of logits, omitting the j'th element
        num_logits = len(logits_list)
        ensemble_target = torch.zeros(logits_list[j].shape).to(self.device)
        for i, logits in enumerate(logits_list):
            if i != j:
                ensemble_target += (1 / (num_logits - 1)) * F.softmax(logits, dim=-1)
        return ensemble_target

    def train_students(
        self,
        epochs=20,
        plot_losses=True,
        save_model=True,
        save_model_path="./Experiments",
    ):
        for student in self.student_cohort:
            student.train()

        loss_arr = []

        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_student_model_weights = deepcopy(self.student_cohort[0].state_dict())
        self.best_student = self.student_cohort[0]
        self.best_student_id = 0
        num_students = len(self.student_cohort)

        print("\nTraining students...")

        for ep in tqdm(range(epochs), position=0):
            epoch_loss = 0.0
            correct = 0
            cohort_ce_loss = [0 for s in range(num_students)]
            cohort_divergence = [0 for s in range(num_students)]
            cohort_entropy = [0 for s in range(num_students)]

            epoch_len = int(len(self.train_loader.dataset) / self.train_loader.batch_size)

            for (data, label) in tqdm(self.train_loader, total=epoch_len, position=1):

                data = data.to(self.device)
                label = label.to(self.device)

                for optim in self.student_optimizers:
                    optim.zero_grad()
                
                # Forward passes to compute logits
                student_outputs = []
                for n in self.student_cohort:
                    logits = n(data)
                    student_outputs.append(logits)

                avg_student_loss = 0
                
                for i in range(num_students):
                    student_loss = 0
                    # Calculate ensemble target
                    target = self.ensemble_target(student_outputs, i)
                    student_loss += self.loss_fn(student_outputs[i], target.detach())
                    ce_loss = F.cross_entropy(student_outputs[i], label)

                    # running average of both loss summands
                    cohort_ce_loss[i] += (1 / epoch_len) * ce_loss
                    cohort_divergence[i] += (1 / epoch_len) * student_loss

                    # running average of output entropy
                    output_distribution = Categorical(logits=student_outputs[i])
                    entropy = output_distribution.entropy().mean(dim=0)
                    cohort_entropy[i] += (1 / epoch_len) * entropy
                     
                    student_loss += ce_loss
                    student_loss.backward()
                    self.student_optimizers[i].step()

                    avg_student_loss += (1 / num_students) * student_loss

                predictions = []
                correct_preds = []
                for i in range(num_students):
                    predictions.append(student_outputs[i].argmax(dim=1, keepdim=True))
                    correct_preds.append(
                        predictions[i].eq(label.view_as(predictions[i])).sum().item()
                    )

                correct += sum(correct_preds) / len(correct_preds)

                epoch_loss += avg_student_loss

            epoch_acc = correct / length_of_dataset
            # TODO log training accuracy for each student separately

            for student_id, student in enumerate(self.student_cohort):
                _, epoch_val_acc = self._evaluate_model(student, verbose=False)

                if epoch_val_acc > best_acc:
                    best_acc = epoch_val_acc
                    self.best_student_model_weights = deepcopy(student.state_dict())
                    self.best_student = student
                    self.best_student_id = student_id
                
                if self.log:
                    self.writer.add_scalar("Accuracy/Validation student"+str(student_id), epoch_val_acc, ep)
                    self.writer.add_scalar("Loss/Cross-entropy student"+str(student_id), cohort_ce_loss[student_id], ep)
                    self.writer.add_scalar("Loss/Divergence student"+str(student_id), cohort_divergence[student_id], ep)
                    self.writer.add_scalar("Loss/Entropy student"+str(student_id), cohort_entropy[student_id], ep)

            if self.log:
                self.writer.add_scalar("Loss/Train average", epoch_loss, ep)
                self.writer.add_scalar("Accuracy/Train average", epoch_acc, ep)

            loss_arr.append(epoch_loss)
            
            if self.student_schedulers:
                for i in range(num_students):
                    self.student_schedulers[i].step()
                
                if ep % 10 == 0:
                    print(f"Epoch {ep}, Learning rate {self.student_schedulers[i].get_last_lr()}")
                    print(f"Epoch: {ep}, Loss: {epoch_loss}, Training accuracy: {epoch_acc}")

        self.best_student.load_state_dict(self.best_student_model_weights)
        if save_model:
            print(f"The best student model is the model number {self.best_student_id} in the cohort")
            if not os.path.isdir(save_model_path):
                os.mkdir(save_model_path)
            torch.save(self.best_student.state_dict(), os.path.join(
                save_model_path, ("student" + str(self.best_student_id) + ".pt")))
        if plot_losses:
            plt.plot(loss_arr)

    def _evaluate_model(self, model, verbose=True):
        """
        Evaluate the given model's accuaracy over val set.
        For internal use only.

        :param model (nn.Module): Model to be used for evaluation
        :param verbose (bool): Display Accuracy
        """
        model.eval()
        length_of_dataset = len(self.val_loader.dataset)
        correct = 0
        outputs = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)

                if isinstance(output, tuple):
                    output = output[0]
                outputs.append(output)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        if verbose:
            print(f"Accuracy: {correct/length_of_dataset}")

        epoch_val_acc = correct / length_of_dataset
        return outputs, epoch_val_acc

    def evaluate(self):
        """
        Evaluate method for printing accuracies of the trained student networks

        """

        for i, student in enumerate(self.student_cohort):
            print("-" * 80)
            model = deepcopy(student).to(self.device)
            print(f"Evaluating student {i}")
            out, acc = self._evaluate_model(model)

    def get_parameters(self):
        """
        Get the number of parameters for the teacher and the student network
        """

        print("-" * 80)
        for i, student in enumerate(self.student_cohort):
            student_params = sum(p.numel() for p in student.parameters())
            print(f"Total parameters for the student network {i} are: {student_params}")

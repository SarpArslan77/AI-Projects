
# train.py

#! Custom TODO notes:
#TODO AD: Add docstring.
#TODO ATH: Add type hint.
#TODO CTH: Check type hint.
#TODO FTH: Fix the hardcoding.
#TODO HPE: Handle possible error.
#TODO AC: Add comment.
#TODO AIC: Add input control.

#! PW: Possibly wrong.

from dataclasses import dataclass
from datetime import datetime
from typing import Any
import os

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from lstm import LSTM

@dataclass
class ConfigTrainer:
    dataloaders: tuple[DataLoader, DataLoader, DataLoader]
    learning_rate: float
    num_epochs: int
    max_norm: float
    print_freq: int
    val_freq: int
    save_checkpoint_freq: int
    model_save_path: str
    saved_checkpoint: dict[str, Any] | None = None
    show_graph: bool = True

    def __post_init__(self) -> None:
        # - learning_rate
        if not isinstance(self.learning_rate, float):
            raise TypeError(f"learning_rate must be an float, got {type(self.learning_rate).__name__}.")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}.")

        # - num_epochs
        if not isinstance(self.num_epochs, int):
            raise TypeError(f"num_epochs must be an integer, got {type(self.num_epochs).__name__}.")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}.")

        # - max_norm
        if not isinstance(self.max_norm, float):
            raise TypeError(f"max_norm must be an float, got {type(self.max_norm).__name__}.")
        if self.max_norm <= 0:
            raise ValueError(f"max_norm must be positive, got {self.max_norm}.")

        # - print_freq
        if not isinstance(self.print_freq, int):
            raise TypeError(f"print_freq must be an integer, got {type(self.print_freq).__name__}.")
        if self.print_freq < 0:
            raise ValueError(f"print_freq must be non-negative, got {self.print_freq}.")

        # - val_freq
        if not isinstance(self.val_freq, int):
            raise TypeError(f"val_freq must be an integer, got {type(self.val_freq).__name__}.")
        if self.val_freq < 0:
            raise ValueError(f"val_freq must be non-negative, got {self.val_freq}.")

        # - save_checkpoint_freq
        if not isinstance(self.save_checkpoint_freq, int):
            raise TypeError(f"save_checkpoint_freq must be an integer, got {type(self.save_checkpoint_freq).__name__}.")
        if self.save_checkpoint_freq < 0:
            raise ValueError(f"save_checkpoint_freq must be non-negative, got {self.save_checkpoint_freq}.")

        # - model_save_path
        if not isinstance(self.model_save_path, str):
            raise TypeError(f"model_save_path must be a string, got {type(self.model_save_path).__name__}.")        
        if not os.path.exists(self.model_save_path):
            raise FileNotFoundError(f"model_save_path does not exist: {self.model_save_path}")
        if not os.path.isdir(self.model_save_path):
            raise NotADirectoryError(f"model_save_path exists but is not a directory: {self.model_save_path}")
        
        # - saved_checkpoint
        if self.saved_checkpoint:
            if not isinstance(self.saved_checkpoint, dict):
                raise TypeError(f"saved_checkpoint must be an dict[str, Any], got {type(self.saved_checkpoint).__name__}.")

        # - show_graph
        if not isinstance(self.show_graph, bool):
            raise TypeError(f"show_graph must be a boolean, got {type(self.show_graph).__name__}.")   

# Define the types.
history_type = list[Tensor]

class Trainer():
    def __init__(
            self,
            config_trainer: ConfigTrainer,
            model: LSTM
    ) -> None:
        # Define the input parameters.
        self.cfg_t: ConfigTrainer = config_trainer
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.cfg_t.dataloaders

        # Define the setup objects.
        self.model = model

        self.loss_func = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(
            params = self.model.parameters(),
            lr = self.cfg_t.learning_rate
        )

        self.current_epoch: int = 0
        self.trial_folder_path: str | None = None

        # Create trackers for the training and validation, accuracies and average losses.
        self.train_acc_history: list[float] = []
        self.train_avg_loss_history: list[float] = []
        self.val_acc_history: list[float] = []
        self.val_avg_loss_history: list[float] = []

        # If an previously started training is being continued, import the parameters.
        if self.cfg_t.saved_checkpoint: 
            # Unpack the parameters from the checkpoint.
            self.current_epoch = self.cfg_t.saved_checkpoint["epoch"]

            self.model.load_state_dict(self.cfg_t.saved_checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(self.cfg_t.saved_checkpoint["optimizer_state_dict"])

            self.train_acc_history, self.train_avg_loss_history = self.cfg_t.saved_checkpoint["train_history"]
            self.val_acc_history, self.val_avg_loss_history = self.cfg_t.saved_checkpoint["val_history"]

    def _train_one_epoch(self) -> None:
        # 1. Open the training mode.
        self.model.train()

        # 2. Check for the print frequencies.
        print_this_epoch: bool = ((self.current_epoch+1) % self.cfg_t.print_freq == 0)

        # 3. Create trackers for the training predictions and targets.
        train_logits: list[Tensor] = []
        train_targets: list[Tensor] = []
        train_losses: list[Tensor] = []

        # 4. Iterate over the training dataset.
        for i, (train_input, train_target) in enumerate(self.train_dataloader):
            # 4.1. Forward pass.
            train_logit: Tensor = self.model(train_input)
            train_loss: Tensor = self.loss_func(train_logit, train_target)

            # 4.2. Backward pass and optimize.
            self.optimizer.zero_grad()
            train_loss.backward()

            # 4.3. Gradient Clipping.
            # This prevents gradients from getting too large.
            torch.nn.utils.clip_grad_norm_(
                parameters = self.model.parameters(),
                max_norm = self.cfg_t.max_norm
            )

            self.optimizer.step()

            # 4.4. Track the variables for this epoch.
            train_logits.append(train_logit.detach())
            train_targets.append(train_target)
            train_losses.append(train_loss.item())

        # 5. Calculate the mean of training losses.
        train_avg_loss: float = sum(train_losses) / len(train_losses)

        # 6. Calculate the training accuracy.
        train_acc: float = self._calculate_accuracy(
            logits = train_logits,
            targets = train_targets
        )

        if print_this_epoch:
            print(f"\nEpoch: {self.current_epoch+1}/{self.cfg_t.num_epochs}") 
            print(f" -> Training Accuracy: {(train_acc*100):.2f} % | Loss: {train_avg_loss:.4f}")

        # 8. Add both the training- accuracy and loss into history trackers.
        self.train_acc_history.append(train_acc)
        self.train_avg_loss_history.append(train_avg_loss)

    def _validate_one_epoch(self) -> None:
        with torch.no_grad():
            # 1. Open the evaluation mode for the validation.
            self.model.eval()

            # 2. Create trackers for the validation prediction and targets.
            val_logits: list[Tensor] = []
            val_targets: list[Tensor] = []
            val_losses: list[Tensor] = []

            # 3. Iterate over the validation dataset.
            for i, (val_input, val_target) in enumerate(self.val_dataloader):
                # 3.1. Forward pass.
                val_logit: Tensor = self.model(val_input)
                val_loss: Tensor = self.loss_func(val_logit, val_target)

                # 3.2. Track the variables for this validation run.
                val_logits.append(val_logit.detach())
                val_targets.append(val_target)
                val_losses.append(val_loss.item())

            # 4. Calculate the mean of validation losses.
            val_avg_loss: float = sum(val_losses) / len(val_losses)

            # 5. Calculate the accuracy.
            val_acc: float = self._calculate_accuracy(
                logits = val_logits,
                targets = val_targets
            )

            # 6. Track the validation accuracy and loss.
            self.val_avg_loss_history.append(val_loss)
            self.val_acc_history.append(val_acc)

            print(f" -> Validation Accuracy: {(val_acc*100):.2f} % | Loss: {val_avg_loss:.4f}")

    def _save_checkpoint(self) -> None:
        # 1. Create the checkpoint path.
        checkpoint_path: str = os.path.join(self.trial_folder_path, f"model_epoch_{self.current_epoch+1}.pth")

        # 2. Pack all the accuracy and loss tracker lists into history tuples.
        train_history: tuple[history_type, history_type] = (self.train_acc_history, self.train_avg_loss_history)
        val_history: tuple[history_type, history_type] = (self.val_acc_history, self.val_avg_loss_history)

        # 3. Create a dictionary to hold all the states.
        checkpoint: dict[str, Any] = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_history": train_history,
            "val_history": val_history
        }

        # 4. Save the checkpoint dictionary.
        torch.save(
            obj = checkpoint,
            f = checkpoint_path
        )

    def _create_folder(
            self,
            train_start_time: datetime
    ) -> None:
        trial_folder_name: str = train_start_time.strftime("%Y-%m-%d_%H-%M")
        self.trial_folder_path = os.path.join(self.cfg_t.model_save_path, trial_folder_name)

        # Try to create a folder for the new trial. #TODO Change it so that even though the creation is failed, the program still runs.
        try:
            os.makedirs(self.trial_folder_path)
            print(f"\nDirectory {self.trial_folder_path} has been created succesfully!")
        except PermissionError:
            print("\nNew directory creation doesn't have the necessary permissions to write to that file.")
        except FileExistsError:
            print("\nA Directory with an exact name exists.")


    def _calculate_accuracy( #TODO Try diffferent type of parameters then binary clasisiciation.
            self,
            logits: list[Tensor],
            targets: list[Tensor]
    ) -> float:
        # 1. Combine th logit and target lists into one long tensors.
        all_logits: Tensor = torch.cat(logits)
        all_targets: Tensor = torch.cat(targets)

        # 2. In order to interpret the results as an accuracy value, the logits has to be converted into probabilities.
        # 2.1. Convert the raw logits into probabilities [0.0, 1.0].
        probs: Tensor = torch.sigmoid(all_logits)
        # 2.2. Convert probabilities into integers (> 0.5 becomes 1, else 0).
        preds: Tensor = torch.round(probs)
        # 2.3. Compare the integers to the targets and calculate the accuracy.
        matches: Tensor = (preds == all_targets)
        num_correct: Tensor = torch.sum(matches)
        acc: float = (num_correct / len(preds)).item()

        return acc
    
    def test(self) -> None:
        print("\nStarting with testing!")

        with torch.no_grad():
            # 1. Open the evaluation mode for the testing.
            self.model.eval()

            # 2. Create the trackers for the test predictions and targets of each test run.
            test_logits: list[Tensor] = []
            test_targets: list[Tensor] = []

            # 3. Iterate over the test dataset.
            for i, (test_input, test_target) in enumerate(self.test_dataloader):
                # 3.1. Forward pass.
                test_logit: Tensor = self.model(test_input)

                # 3.2. Track the variables.
                test_logits.append(test_logit.detach())
                test_targets.append(test_target)

            # 4. Calculate the accuracy.
            test_acc: float = self._calculate_accuracy(
                logits = test_logits,
                targets = test_targets
            )

            print(f" ---> Test Accuracy: {(test_acc*100):.2f} %")

    def fit(
            self,
            test_after_training: bool = True
    ) -> tuple[tuple[history_type, history_type], tuple[history_type, history_type], tuple[float, float, float]]:
        # 1. Start the training.
        print("\nStarting with Training!")

        # 2. Start tracking the training time.
        train_start_time = datetime.now()

        # 3. Create a timestamped folder.
        self._create_folder(train_start_time=train_start_time)

        # 4. Start with the loop.
        for epoch in range(self.cfg_t.num_epochs):
            # 5. Update the current epoch of this class instance.
            self.current_epoch = epoch

            # 6. Train the model for one epoch.
            self._train_one_epoch()

            # 7. Check whether validation should take place in this epoch.
            validate_this_epoch: bool = ((epoch+1) % self.cfg_t.val_freq == 0)

            if validate_this_epoch:
                self._validate_one_epoch()

            # 8. Check whether the checkpoint save should take place in this epoch.
            save_checkpoint_this_epoch: bool = ((epoch+1) % self.cfg_t.save_checkpoint_freq == 0)
            if save_checkpoint_this_epoch:
                self._save_checkpoint()

        # 9. Calculate the training time.
        train_end_time = datetime.now()
        train_elapsed_secs: float = (train_end_time - train_start_time).total_seconds()
        train_elapsed_mins, train_sec = divmod(train_elapsed_secs, 60)
        train_hour, train_min = divmod(train_elapsed_mins, 60)
        # 9.1 Pack all the times in a tuple for the title of the graph.
        train_time: tuple[float, float, float] = (train_hour, train_min, train_sec)
        print(f"\nFinished Training!\nTraining Time: {int(train_hour):02d}:{int(train_min):02d}:{int(train_sec):02d}")

        # 10. Convert all the lists into numpy arrays.
        train_acc_history_np: NDArray = np.array(self.train_acc_history)
        train_avg_loss_history_np: NDArray = np.array(self.train_avg_loss_history)
        val_acc_history_np: NDArray = np.array(self.val_acc_history)
        val_avg_loss_history_np: NDArray = np.array(self.val_avg_loss_history)

        # 11. Pack all the accuracy and loss tracker numpy arrays into history tuples.
        acc_history: tuple[history_type, history_type] = (train_acc_history_np, val_acc_history_np)
        avg_loss_history: tuple[history_type, history_type] = (train_avg_loss_history_np, val_avg_loss_history_np)

        # 12. If requested, test the model after the training.
        if test_after_training:
            self.test()

        # 13. Return the history trackers for plotting them.
        return (acc_history, avg_loss_history, train_time)
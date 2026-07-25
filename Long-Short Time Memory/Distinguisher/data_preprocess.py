
# data_preprocess.py

#! Custom TODO notes:
#TODO AD: Add docstring.
#TODO ATH: Add type hint.
#TODO CTH: Check type hint.
#TODO FTH: Fix the hardcoding.
#TODO HPE: Handle possible error.
#TODO AC: Add comment.
#TODO AIC: Add input control.

from dataclasses import dataclass
import secrets

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.utils.data import (
    TensorDataset,
    random_split,
    DataLoader
)

@dataclass
class ConfigDataManager:
    # Generator Settings.
    num_count: int
    multiplier: int
    increment: int
    modulus: int
    weak_nums_seed: int
    strong_num_split: float

    # Preprocessing Settings.
    sequence_length: int
    stride: int
    batch_size: int
    train_split: float
    val_split: float

    def __post_init__(self) -> None: # This method runs on it own, after the initialization.
        # - num_count
        if not isinstance(self.num_count, int):
            raise TypeError(f"num_count must be an integer, got {type(self.num_count).__name__}.")
        if self.num_count <= 0:
            raise ValueError(f"num_count must be positive, got {self.num_count}.")

        # - sequence_length
        if not isinstance(self.sequence_length, int):
            raise TypeError(f"sequence_length must be an integer, got {type(self.sequence_length).__name__}.")
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {self.sequence_length}.")

        # - multiplier
        if not isinstance(self.multiplier, (int, float)):
            raise TypeError(f"multiplier must be a number, got {type(self.multiplier).__name__}")

        # - increment
        if not isinstance(self.increment, (int, float)):
            raise TypeError(f"increment must be a number, got {type(self.increment).__name__}")

        # - modulus
        if not isinstance(self.modulus, int):
            raise TypeError(f"modulus must be an integer, got {type(self.modulus).__name__}.")
        if self.modulus <= 0:
            raise ValueError(f"modulus must be positive, got {self.modulus}.")

        # - weak_nums_seed
        if not isinstance(self.weak_nums_seed, int):
            raise TypeError(f"weak_nums_seed must be a number, got {type(self.weak_nums_seed).__name__}")

        # - stride
        if not isinstance(self.stride, int):
            raise TypeError(f"stride must be an integer, got {type(self.stride).__name__}.")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}.")

        # - strong_num_percentage
        if not isinstance(self.strong_num_split, float):
            raise TypeError(f"strong_num_percentage must be a float, got {type(self.strong_num_split).__name__}.")
        if not (0.0 <= self.strong_num_split <= 1.0):
            raise ValueError(f"strong_num_percentage must be between 0.0 and 1.0, got {self.strong_num_split}.")

        # - train_split
        if not isinstance(self.train_split, float):
            raise TypeError(f"train_split must be a float, got {type(self.train_split).__name__}.")
        if not (0.0 <= self.train_split <= 1.0):
            raise ValueError(f"train_split must be between 0.0 and 1.0, got {self.train_split}.")

        # - val_split
        if not isinstance(self.val_split, float):
            raise TypeError(f"val_split must be a float, got {type(self.val_split).__name__}.")
        if not (0.0 <= self.val_split <= 1.0):
            raise ValueError(f"val_split must be between 0.0 and 1.0, got {self.val_split}.")

        # --- train_split and val_split
        if (self.train_split + self.val_split) > 1:
            raise ValueError(f"train_split + val_split must be smaller then 1, got {self.train_split + self.val_split}")

        # - batch_size
        if not isinstance(self.batch_size, int):
            raise TypeError(f"batch_size must be an integer, got {type(self.batch_size).__name__}.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")

class DataManager():
    def __init__(
            self,
            config_data_manager: ConfigDataManager
    ) -> None:
        # Define the input parameters.
        self.cfg_dm: ConfigDataManager = config_data_manager

        # Initialize the data loaders.
        self.train_dataloader: DataLoader | None = None
        self.val_dataloader: DataLoader | None = None
        self.test_dataloader: DataLoader | None = None
        
    def _weak_num_create(
            self,
            x: int
    ) -> int:
        """
        Generates the next pseudo-random number using a Linear Congrutential Generator (LCG).

        Implements the formula: x_next = (multiplier * x + increment) % modulus.

        Args:
            x (int): The current state (seed or previous number) of the generator.
        
        Returns:
            int: The next pseudo-random integer in the sequence.
        """

        return (self.cfg_dm.multiplier * x + self.cfg_dm.increment) % self.cfg_dm.modulus

    def _create_sequence(
            self,
            data: NDArray
    ) -> NDArray:
        """
        Generates overlapping sequences.

        Args:
            data (NDArray): The input time-series data.
        
        Returns:
            NDArray: A 2D array of shape (num_sequences, sequence_length).
        """  

        # Creates a sliding window view of data.
        windows: NDArray = np.lib.stride_tricks.sliding_window_view(
            x = data,
            window_shape = self.cfg_dm.sequence_length
        )

        # Apply the stride step to select specific windows.
        strided_windows: NDArray = windows[::self.cfg_dm.stride] # Syntax: array[start:stop:step]

        return strided_windows

    def _generate_synthetic_data(self) -> tuple[NDArray, NDArray]:
        """
        Generates synthetic strong and weak number sequences with their corresponding targets.

        Returns:
            Tuple[NDArray,NDArray]: A tuple containing the shuffled feature sequences and the reshaped target vector.
        """
        
        # 1. Define the amount of numbers from the percentage.
        strong_num_count: int = int(self.cfg_dm.num_count * self.cfg_dm.strong_num_split)
        weak_num_count: int = self.cfg_dm.num_count - strong_num_count

        strong_nums: list[int] = [secrets.randbelow(self.cfg_dm.modulus) for _ in range(strong_num_count)]
        weak_nums: list[int] = [self.cfg_dm.weak_nums_seed]
        for _ in range(weak_num_count-1): # '-1' due to the starting seed.
            weak_num: int = self._weak_num_create(x=weak_nums[-1])
            weak_nums.append(weak_num)

        # Convert the lists into numpy arrays.
        strong_nums_np: NDArray = np.array(object=strong_nums)
        weak_nums_np: NDArray = np.array(object=weak_nums)
        # Enforce, that the arrays have float32 type, which is required for the training.
        strong_nums_float32: NDArray = strong_nums_np.astype(dtype=np.float32)
        weak_nums_float32: NDArray = weak_nums_np.astype(dtype=np.float32)
        # Normalize the arrays.
        strong_nums_normalized: NDArray = strong_nums_float32 / self.cfg_dm.modulus
        weak_nums_normalized: NDArray = weak_nums_float32 / self.cfg_dm.modulus

        # 2. Create the data sequences.
        strong_num_sequences: NDArray = self._create_sequence(
            data = strong_nums_normalized
        )
        weak_num_sequences: NDArray = self._create_sequence(
            data = weak_nums_normalized
        )

        # 3. Create the targets.
        strong_targets: NDArray = np.ones(shape=(len(strong_num_sequences), 1))
        weak_targets: NDArray = np.zeros(shape=(len(weak_num_sequences), 1))

        # 4. Merge the data and targets.
        num_sequences_np: NDArray = np.concatenate(
            (strong_num_sequences, weak_num_sequences), # These have to be numpy arrays.
            axis = 0
        )
        targets_np: NDArray = np.concatenate(
            (strong_targets, weak_targets),
            axis = 0
        )

        # Enforce, that the targets have the type float32.
        targets_float32: NDArray = targets_np.astype(dtype=np.float32)

        # Create prermutation indices.
        permutation_indices: NDArray = np.random.permutation(x=num_sequences_np.shape[0]) # Shuffle the rows.

        # Apply the permutation to both arrays.
        nums_shuffled: NDArray = num_sequences_np[permutation_indices, :] # (batch_size, sequence_length)
        targets_shuffled: NDArray = targets_float32[permutation_indices] # (batch_size)

        # Reshape the targets by adding a second dimenstion, so it becomes a column vector.
        targets_reshaped: NDArray = targets_shuffled.reshape(-1, 1) # '-1' calculates the number of rows automatically.

        return (nums_shuffled, targets_reshaped)
    
    def _create_dataloaders(
            self,
            nums: NDArray,
            targets: NDArray
    ) -> None:
        """
        Converts raw numpy arrays into PyTorch tensors and populates the data loaders.

        Splits the dataset into training, validation, and testing sets based on 
        configured ratios.

        Args:
            nums (NDArray): The feature sequences.
            targets (NDArray): The target labels.
        """

        # 1. Create tensors for the data and targets.
        nums_tensor: Tensor = torch.from_numpy(nums)
        targets_tensor: Tensor = torch.from_numpy(targets)

        # Reshape the numbers tensor, so it works with the LSTM forwarding.
        nums_reshaped: Tensor = nums_tensor.unsqueeze(dim=-1) # '-1' adds a new dimension to the last position.
        # (batch_size, sequence_length, input_features)

        # 2. Seperate the data into training, validation and test sets.
        # Bind the numbers and its targets into one single tensor, so they get seperated together later on.
        dataset = TensorDataset(nums_reshaped, targets_tensor)

        # Calculate the split sizes.
        dataset_size: int = nums_reshaped.shape[0]
        train_size: int = int(dataset_size * self.cfg_dm.train_split)
        val_size: int = int(dataset_size * self.cfg_dm.val_split)
        test_size: int = dataset_size - train_size - val_size
        
        # Split the dataset.
        train_dataset, val_dataset, test_dataset = random_split(
            dataset = dataset,
            lengths = [train_size, val_size, test_size]
        )

        # 3. Create the dataloaders.
        self.train_dataloader = DataLoader(
            dataset = train_dataset,
            batch_size = self.cfg_dm.batch_size,
            shuffle = True
        )
        self.val_dataloader = DataLoader(
            dataset = val_dataset,
            batch_size = self.cfg_dm.batch_size,
            shuffle = False
        )
        self.test_dataloader = DataLoader(
            dataset = test_dataset,
            batch_size = self.cfg_dm.batch_size,
            shuffle = False
        )

    def prepare_data(
            self
    ) -> None:
        """
        Orchestrates the data pipeline by generating synthetic data and creating loaders.
        """
        
        nums, targets = self._generate_synthetic_data()

        self._create_dataloaders(
            nums = nums,
            targets = targets
        )

    def get_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Returns the loaders. Raises error if data is not prepared.
        """

        if self.train_dataloader is None:
            raise RuntimeError("Data has not been prepared yet. Run prepare_dat() first.")
        
        return (self.train_dataloader, self.val_dataloader, self.test_dataloader)
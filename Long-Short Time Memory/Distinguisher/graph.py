
# graph.py

#! Custom TODO notes:
#TODO AD: Add docstring.
#TODO ATH: Add type hint.
#TODO CTH: Check type hint.
#TODO FTH: Fix the hardcoding.
#TODO HPE: Handle possible error.
#TODO AC: Add comment.
#TODO AIC: Add input control.

from dataclasses import dataclass
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import mplcursors
import numpy as np
from numpy.typing import NDArray

@dataclass
class ConfigHistoryPlotter:
    num_epochs: int
    train_time: tuple[float, float, float]

    def __post_init__(self) -> None: 
        # Unpack the times.
        train_hour, train_min, train_sec = self.train_time
        
        # - num_epochs
        if not isinstance(self.num_epochs, int):
            raise TypeError(f"num_epochs must be an integer, got {type(self.num_epochs).__name__}.")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}.")
        
        # - train_hour
        if not isinstance(train_hour, float):
            raise TypeError(f"train_hour must be an integer, got {type(train_hour).__name__}.")
        if train_hour < 0:
            raise ValueError(f"train_hour must be non-negative, got {train_hour}.")
        # - train_min
        if not isinstance(train_min, float):
            raise TypeError(f"train_min must be an integer, got {type(train_min).__name__}.")
        if train_min < 0:
            raise ValueError(f"train_min must be non-negative, got {train_min}.")
        # - train_sec
        if not isinstance(train_sec, float):
            raise TypeError(f"train_sec must be an integer, got {type(train_sec).__name__}.")
        if train_sec < 0:
            raise ValueError(f"train_sec must be non-negative, got {train_sec}.")

# Define the variable types hint.
history_type = tuple[NDArray, NDArray]

class HistoryPlotter():
    def __init__(
        self,
        config_history_plotter: ConfigHistoryPlotter,
    ) -> None:
        # Define the input parameters.
        self.cfg_ht: ConfigHistoryPlotter = config_history_plotter

        # Unpack the training time for the title of whole figure.
        train_hour, train_min, train_sec = self.cfg_ht.train_time

        # Define the figure.
        self.fig: Figure = plt.figure(figsize=(10, 4))
        self.fig.suptitle(f"Training Time: {int(train_hour):02d}:{int(train_min):02d}:{int(train_sec):02d}")

        # Define the gridspace.
        self.gs = gridspec.GridSpec(
            nrows = 1,
            ncols = 2,
            figure = self.fig
        )
        # Adjust the spacing between the subplots.
        self.gs.update(wspace=0.3)
    
    def _plot_subplot(
            self,
            ax: Axes,
            data: tuple[NDArray, NDArray],
            colors: tuple[str, str],
            linestyle: str,
            label: str,
    ) -> None:
        # 1. Split the parameters into training and validation.
        train_data, val_data = data
        train_color, val_color = colors

        # 2. Extend the validation data, so it's size matches the training data.
        # Get the number of training and validation samples.
        n_train_data: int = len(train_data)
        n_val_data: int = len(val_data)

        # Find out the repeating frequency and remainder.
        freq: int = n_train_data // n_val_data
        remain: int = n_train_data % n_val_data

        # Initialize every element with freq.
        freq_per_val_data = np.full(
            shape = n_val_data,
            fill_value = freq
        )

        # Add the remainder to the last element.
        freq_per_val_data[n_val_data-remain:] += 1

        # Perform the repetition.
        expanded_val_data: NDArray = np.repeat(
            a = val_data,
            repeats = freq_per_val_data
        )

        # 3. Create the x-axis vector. 
        num_epochs_vector: NDArray = np.arange(
            start = 1,
            stop = self.cfg_ht.num_epochs + 1
        )

        # 4. Plot the training data.
        train_line, = ax.plot(
            num_epochs_vector,
            train_data,
            color = train_color,
            label = f"Training {label}",
            linestyle = linestyle
        )

        # 5. Plot the validation data.
        val_line, = ax.plot(
            num_epochs_vector,
            expanded_val_data,
            color = val_color,
            label = f"Validation {label}",
            linestyle = linestyle
        )

        # 6. Set the settings for the axis.
        ax.set_title(f"{label} History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Training- and Validation {label}")
        ax.grid(visible=True)
        ax.legend(handles = [train_line, val_line])

        # 7. Add a cursor to the graph.
        mplcursors.cursor([train_line, val_line])

    def _save_plots(
            self,
            trial_folder_path: str
    ) -> None:
        try: # Try to create a graph for the accuracy and loss.
            graph_path: str = os.path.join(trial_folder_path, "acc_and_loss_history.png")
            plt.savefig(graph_path)
            print(f"\nThe Graph {graph_path} has been created succesfully!")
        except PermissionError:
            print("\nThe Graph doesn't have the necessary permissions to write to that file...")
        except Exception as e:
            print(f"\nThe saving of ratio graph has failed due to the error: {e}")

    def plot_history(
            self,
            histories: tuple[history_type, history_type],
            trial_folder_path: str,
            show_graph: bool = True
    ) -> None:
        # 1. Unpack the histories.
        acc_history, avg_loss_history = histories

        # 2. Define the axes.
        acc_ax: Axes = self.fig.add_subplot(self.gs[0, 0])
        avg_loss_ax: Axes = self.fig.add_subplot(self.gs[0, 1])

        # 3. Plot the accuracy.
        self._plot_subplot(
            ax = acc_ax,
            data = acc_history,
            colors = ("red", "blue"),
            linestyle = "-",
            label = "Accuracy"
        )

        # 4. Plot the average loss.
        self._plot_subplot(
            ax = avg_loss_ax,
            data = avg_loss_history,
            colors = ("red", "blue"),
            linestyle = "--",
            label = "Average Loss"
        )

        # 5. Show the legends.
        acc_ax.legend()
        avg_loss_ax.legend()

        # 6. Save the graphs.
        self._save_plots(trial_folder_path=trial_folder_path)

        # 7. If requested, show the graphs.
        if show_graph:
            plt.show()

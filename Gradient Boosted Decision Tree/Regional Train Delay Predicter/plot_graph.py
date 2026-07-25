
#* plot_graph.py

#* ======= Library Imports =======
# ------- Native Library. -------
from dataclasses import dataclass
from typing import Any

# ------- External Libraries. -------
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame
from pathlib import Path
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

#* ======= Config Graph Plotter Dataclass Initialization =======
@dataclass(frozen = True)
class ConfigGraphPlotter:
    """
    Configuration schema for the GraphPlotter visualizer class.

    This dataclass encapsulates subplot layout properties, file export targets,
    aesthetic settings for the classification heatmap, and category labels
    required to configure a unified evaluation figure.

    Attributes:
        subplots_nrows (int): The number of rows in the subplot grid.
        subplots_ncols (int): The number of columns in the subplot grid.
        subplots_figsize (tuple[int, int]): The total size (width, height) of the figure in inches.
        
        current_training_dir (Path): The target folder path where the plot is saved.
        plot_file_name (str): The output filename of the saved plot.
        
        cr_annot (bool): If True, overlay numeric values directly inside the heatmap cells.
        cr_fmt (str): Formatting string for the heatmap's cell annotations (e.g., ".2f").
        cr_annot_kws (dict): Additional text styling arguments (such as font size) for annotations.
        cr_cmap (str): The color scheme designation for the classification heatmap.
        cr_xticklabels (list[str] | bool): Labels for the heatmap x-axis, or True to auto-label using columns.
        cr_yticklabels (list[str] | bool): Labels for the heatmap y-axis, or True to auto-label using index.
        
        cm_display_labels (list[str]): Categorical label list mapped to the confusion matrix axes.
    """

    # ------- Subplot Parameters. -------
    subplots_nrows: int
    subplots_ncols: int
    subplots_figsize: tuple[int, int]

    # ------- File Parameters. -------
    current_training_dir: Path
    plot_file_name: str

    # ------- Heatmap Parameters. -------
    cr_annot: bool
    cr_fmt: str
    cr_annot_kws: dict
    cr_cmap: str
    cr_xticklabels: list[str] | bool
    cr_yticklabels: list[str] | bool

    # ------- Confusion Matrix Parameters. -------
    cm_display_labels: list[str]

#* ======= Graph Plotter Class Initialization =======
class GraphPlotter:
    """
    A utility visualization class to generate, plot, and save model metrics.

    This class initializes a 2x2 grid containing the classifier training loss,
    regressor training loss, classification report heatmap, and confusion matrix,
    ultimately rendering and saving them to disk.

    Attributes:
        cfg (ConfigGraphPlotter): Configuration settings containing dimensions, labels, and file paths.
        fig (plt.Figure): The main matplotlib Figure object housing the subplots.
        axes (NDArray): A 2D NumPy array containing the individual subplot Axes.
    """

    def __init__(
            self,
            config_graph_plotter: ConfigGraphPlotter
    ) -> None:
        """
        Initializes the plotting environment, generates the figure, and configures labels.

        Args:
            config_graph_plotter (ConfigGraphPlotter): Configuration dataclass carrying layout properties 
                and filesystem directories.
        """

        # ------- Initializes class attributes from parameters. -------
        self.cfg: ConfigGraphPlotter = config_graph_plotter

        # ------- Initializes class attributes dynamically. -------
        self.fig, self.axes = plt.subplots(
            self.cfg.subplots_nrows,
            self.cfg.subplots_ncols,
            figsize = self.cfg.subplots_figsize
        )
        self.axes: NDArray

        # ------- Setups the four axises. -------
        # Top-Left for Classifier Training- and Validation Loss.
        classifier_losses_ax: Axes = self.axes[0, 0]

        classifier_losses_ax.set_xlabel(xlabel = "Iterations")
        classifier_losses_ax.set_ylabel(ylabel = "Classifier Training- and Validation Loss")

        # Top-Right for Regressor Training- and Validation Loss.
        regressor_losses_ax: Axes = self.axes[0, 1]

        regressor_losses_ax.set_xlabel(xlabel = "Iterations")
        regressor_losses_ax.set_ylabel(ylabel = "Regressor Training- and Validation Loss")

        # Bottom-Left for Classification Report (CR) Heatmap.
        regressor_cr_ax: Axes = self.axes[1][0]

        regressor_cr_ax.set_xlabel(xlabel = "Metrics")
        regressor_cr_ax.set_ylabel(ylabel = "Classes")
        regressor_cr_ax.set_title(label = "Classification Report Heatmap")

        # Bottom-Right for Confusion Matrix (CM).
        cm_ax: Axes = self.axes[1, 1]

        cm_ax.set_title(label = "Confusion Matrix")

        print("-> Graph Plotter Class initialized!")

    def _plot_loss(
            self,
            loss_vals: tuple[list[float], list[float]],
            ax: Axes
    ) -> None:
        """
        Plots training and validation loss curves over the training iterations.

        Args:
            loss_vals (tuple[list[float], list[float]]): A tuple containing the training loss list (first)
                and validation loss list (second).
            ax (Axes): The target matplotlib subplot Axes where the lines are drawn.
        """

        # Unpacks the losses.
        train_loss, val_loss = loss_vals

        ax.plot(
            train_loss,
            color = "red",
            label = "Train Loss"
        )
        ax.plot(
            val_loss,
            color = "blue",
            linestyle = "--",
            label = "Val Loss"
        )

        # Adds a legend.
        ax.legend()

    def _plot_classification_report_heatmap(
            self,
            classification_report_dict: dict[str, Any]
    ) -> None:
        """
        Converts the raw classification report dictionary into a styled heatmap.

        Args:
            classification_report_dict (dict[str, Any]): Evaluation metrics dictionary from 
                sklearn's classification_report containing precision, recall, and f1-score.
        """

        # ------- Converts the classification report summary to Pandas DataFrame. -------
        report_df = pd.DataFrame(data = classification_report_dict)
        transposed_report_df: DataFrame = report_df.transpose()
        
        # Removes 'accuracy' and 'support' column.
        #? errors = "ignore": Ensures it won't crash if keys are missing.
        simplified_report_df: DataFrame = (
            transposed_report_df
            .drop(
                columns = ["support"], 
                index = ["accuracy"],
                errors = "ignore"
            )
        )

        # ------- Plots the heatmap. -------
        cr_axes: Axes = self.axes[1, 0]

        sns.heatmap(
            simplified_report_df,
            annot = self.cfg.cr_annot,
            fmt = self.cfg.cr_fmt,
            annot_kws = self.cfg.cr_annot_kws,
            cmap = self.cfg.cr_cmap,
            xticklabels = self.cfg.cr_xticklabels,
            yticklabels = self.cfg.cr_yticklabels,
            ax = cr_axes,
        )
        # Rotates the y-axis labels to be horizontal.
        cr_axes.set_yticklabels(
            cr_axes.get_yticklabels(),
            rotation = "horizontal"
        )
    
    def _plot_confusion_matrix(
            self,
            confusion_matrix: NDArray
    ) -> None:
        """
        Generates and draws a normalized confusion matrix visualizer.

        Args:
            confusion_matrix (NDArray): A 2D NumPy array containing target vs predicted 
                classification mappings.
        """

        # ------- Visualizes the Confusion Matrix (CM). -------
        cm_displayer = ConfusionMatrixDisplay(
            confusion_matrix,
            display_labels = self.cfg.cm_display_labels
        )

        cm_ax: Axes = self.axes[1, 1]

        cm_displayer.plot(
            cmap = plt.cm.Blues, #TODO FTW
            ax = cm_ax
        )

        cm_ax.tick_params(
            axis = "x",
            labelrotation = 45
        ) 

    def run_graph_plotter(
            self,
            evals_classifier: dict,
            evals_regressor: dict,
            classification_report_dict: dict[str, Any],
            confusion_matrix: NDArray,
            display_plots: bool
    ) -> None:
        """
        Orchestrates the entire graphing process, exports, and optionally displays the plot.

        This method unpacks classifier and regressor evaluations, plots loss curves, 
        populates the metric heatmap and confusion matrix, adjusts subplot borders,
        saves the unified visual output, and can display it to the screen.

        Args:
            evals_classifier (dict): Training evaluation results from the CatBoost classifier.
            evals_regressor (dict): Training evaluation results from the CatBoost regressor.
            classification_report_dict (dict[str, Any]): Metric details from the final test set evaluations.
            confusion_matrix (NDArray): Normalized class distribution metrics.
            display_plots (bool): Boolean establishing if the matplotlib window should 
                be displayed on runtime execution.
        """

        # ------- Plots the graphs. -------
        # Unpacks the classifier- and regressor loss values.
        classifier_train_loss: list[float] = evals_classifier["learn"]["Logloss"]
        classifier_val_loss: list[float] = evals_classifier["validation"]["Logloss"]

        regressor_train_loss: list[float] = evals_regressor["learn"]["RMSE"]
        regressor_val_loss: list[float] = evals_regressor["validation"]["RMSE"]

        # Plots the classifier- and regressor loss on the top- left and right axises.
        self._plot_loss(
            loss_vals = (classifier_train_loss, classifier_val_loss),
            ax = self.axes[0, 0]
        )

        self._plot_loss(
            loss_vals = (regressor_train_loss, regressor_val_loss),
            ax = self.axes[0, 1]
        )

        # Plots the CR on bottom left axes.
        self._plot_classification_report_heatmap(classification_report_dict = classification_report_dict)

        # Plots the CM on bottom right axes.
        self._plot_confusion_matrix(confusion_matrix = confusion_matrix)

        # ------- Saves and IF needed, displays the plots. -------
        self.fig.tight_layout()

        # Saves the plots.
        plot_dir: Path = self.cfg.current_training_dir / self.cfg.plot_file_name

        plt.savefig(plot_dir)

        if display_plots:
            plt.show()
        
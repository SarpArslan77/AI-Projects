
import torch
from sklearn.metrics import classification_report, confusion_matrix
import intel_extension_for_pytorch as ipex
import matplotlib.pyplot as plt
import matplotlib
import mplcursors
import os
from datetime import datetime
import numpy as np
import seaborn as sns
import torch.nn as nn
import torchvision.transforms as T
import pandas as pd
import json
import torch.optim as optim

from fer_data import (
    test_loader,
    train_loader,
    val_loader,
    standard_pipeline,
)

from cnn import ConvNet

MAIN_FILE_READ_PATH: str = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Facial Emotion Recognition"
json_file_path: str = os.path.join(MAIN_FILE_READ_PATH, "config.json")

# Load the configuration file.
with open(json_file_path, "r") as f:
    config = json.load(f)

# Unpack the parameters into variables for easy acess.
train_cfg = config["training_params"]
early_stop_cfg = config["early_stopping"]
optimizer_cfg = config["optimizer_params"]
scheduler_cfg = config["scheduler_params"]

# Hyperparameters
STARTING_LEARNING_RATE: float = train_cfg["starting_learning_rate"]
num_epochs: int = train_cfg["num_epochs"]
BATCH_SIZE: int = train_cfg["batch_size"]
new_lr: float = train_cfg["starting_learning_rate"]

# Check whether the Intel XE GPU is available, if not use the CPU.
device = torch.device("xpu") if torch.xpu.is_available() else "cpu"
print(f"\nUsing device: {device}") 

# Convolutional Neural Network implementation
model = ConvNet().to(device)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(), 
    lr = STARTING_LEARNING_RATE, 
    weight_decay = optimizer_cfg["weight_decay"],
)

# Scheduler Definition
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode = scheduler_cfg["mode"], # It will look for the metric to stop decreasing.
    factor = scheduler_cfg["factor"], # By which the learning rate will be reduced.
    patience = scheduler_cfg["patience"], # Number of epochs with no improvement after which learning rate will be reduced.
    min_lr = scheduler_cfg["min_lr"], # Minimum learning rate, that the optimizer can drop to.
)

# Apply IPEX optimizations to the model and optimizer
model, optimizer = ipex.optimize(model, optimizer=optimizer)

# Training Loop
print("\nStarting Training!")
training_start_time = datetime.now()
# Tracking parameters for the graph.
train_acc_tracker: list[float] = []
val_acc_tracker: list[float] = []
train_loss_tracker: list[float] = []
val_loss_tracker: list[float] = []
lr_tracker: list[float] = []
# Track the best model with the lowest validation loss.
lowest_val_loss: float = np.inf
best_model_parameters: dict = {}
# Stores the epoch, val_acc and val_loss for the best model, to be marked in the graph.
best_model_val_values: list = [np.nan, np.nan, np.nan] # [epoch, val_acc, val_loss]
for epoch in range(num_epochs):
    print(f"\nEpoch : {epoch+1}/{num_epochs}")
    model.train()
    running_loss: float = 0.0 # Tracking loss for optimizing the learning rate.
    n_correct: int = 0
    n_total: int = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Calculate the accuracy.
        _, predicted = torch.max(outputs.data, 1)
        n_total += labels.size(0)
        # sum() requires the tensor to be on the cpu.
        n_correct += (predicted == labels).cpu().sum().item()
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Update the loss.
        running_loss += loss.item() * images.size(0)

    train_acc: float = 100.0 * n_correct / n_total
    epoch_train_loss: float = running_loss / len(train_loader.dataset)
    print(f"    Train accuracy : {train_acc:.2f}% | Train Loss: {epoch_train_loss:.4f}")
    train_acc_tracker.append(train_acc)
    train_loss_tracker.append(epoch_train_loss)

    # Validation Phase
    model.eval()
    val_loss: float = 0.0
    val_correct: int = 0
    val_total: int = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).cpu().sum().item()
    
    val_acc: float = 100.0 * val_correct / val_total
    epoch_val_loss: float = val_loss / len(val_loader.dataset)
    print(f"    Validation accuracy : {val_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}")
    val_acc_tracker.append(val_acc)
    val_loss_tracker.append(epoch_val_loss)

    # Save the learning rate from the optimizer.
    new_lr = optimizer.param_groups[0]["lr"]
    print(f"    Learning Rate: {new_lr}")
    lr_tracker.append(new_lr)

    # Update the best model and validation loss if the new model is better.
    if epoch_val_loss < lowest_val_loss:
        lowest_val_loss = epoch_val_loss
        best_model_parameters = {
            "epoch" : epoch + 1,
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            "scheduler_state_dict" : scheduler.state_dict(),
            "train_acc_tracker" : train_acc_tracker,
            "train_loss_tracker" : train_loss_tracker,
            "val_acc_tracker" : val_acc_tracker,
            "val_loss_tracker" : val_loss_tracker,
            "lowest_val_loss" : lowest_val_loss,
            "lr_tracker" : lr_tracker,
        }
        best_model_val_values = [epoch, val_acc, epoch_val_loss]

    # Step the scheduler with validation loss
    scheduler.step(epoch_val_loss)
print("\nFinished Training.")
# Calculate the training time.
training_end_time = datetime.now()
elapsed_seconds = (training_end_time - training_start_time).total_seconds()
seconds = elapsed_seconds % 60
minutes = (elapsed_seconds % 3600) // 60
hours = elapsed_seconds // 3600
print(f"\nTotal Training Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

# Define a helper function for graph plotting.
def plot_metric(
    ax: matplotlib.axes.Axes,
    train_tracker: list[float],
    val_tracker: list[float],
    best_epoch: int,
    best_metric: float,
    metric_name: str, 
) -> None:
    # Plot the training and test metric.
    line_train, = ax.plot(
        train_tracker,
        color = "red",
        label = f"Training {metric_name}",
    )
    line_val, = ax.plot(
        val_tracker,
        color = "blue",
        label = f"Validation {metric_name}",
    )

    # Mark the best model's performance with a star.
    ax.plot(
        best_epoch,
        best_metric,
        marker = "*", 
        # marker argument tells the .plot function, which automatically draws lines,
        #   to draw only one point in the graph.
        color = "gold",
        markersize = 12,
        markeredgecolor = "black",
        markeredgewidth = 1,
        label = f"Best Model {metric_name}"
    )

    # Set plot labels and title.
    ax.set_title("Training & Validation Accuracy Progression")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(True)
    ax.legend(
        [line_train, line_val], 
        ["Training Accuracy", "Validation Accuracy"], 
        loc = "lower right",
    )

    # Add interactive cursor
    mplcursors.cursor(line_train)
    mplcursors.cursor(line_val)

# Create a graph for tracking the accuracy, loss and learning rate.
fig, ((ax_acc, ax_loss) , (ax_cr, ax_cm)) = plt.subplots(2, 2, figsize=(10, 10))

# Unpack the best model values for the graph marking.
best_epoch = best_model_val_values[0]
best_acc = best_model_val_values[1]
best_loss = best_model_val_values[2]

# Use the helper function for plotting accuracy and loss.
plot_metric(
    ax_acc,
    train_acc_tracker,
    val_acc_tracker,
    best_epoch,
    best_acc,
    "Accuracy",
)
plot_metric(
    ax_loss,
    train_loss_tracker,
    val_loss_tracker,
    best_epoch,
    best_loss,
    "Loss",
)

emotion_labels: list[str] = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
# Track all the model predictions and correct labels.
all_predictions_gpu: list = []
all_labels_gpu: list = []

# Testing loop
model.eval()
print("\nStarting Testing!")
with torch.no_grad():
    n_correct: int = 0
    n_samples: int = 0
    for j, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        # Max returns (value, index)
        _, predicted = torch.max(outputs.data, 1)

        # Append the predictions and labels from this batch to our master lists.
        all_predictions_gpu.append(predicted)
        all_labels_gpu.append(labels)

        n_samples += labels.size(0)
        n_correct += (predicted == labels).cpu().sum().item()

    # Concatenate and move to CPU.
    final_predictions_cpu = torch.cat(all_predictions_gpu).cpu().numpy()
    final_labels_cpu = torch.cat(all_labels_gpu).cpu().numpy()

    accuracy: float = n_correct / n_samples * 100.0
    print(f"\nAccuracy of the network is: {accuracy:.2f} %")

    # Define a helper function for plotting the heatmaps.
    def plot_heatmap(
        data,
        annot: bool,
        fmt: str,
        annot_kws: dict,
        cmap: str,
        xticklabels: list[str],
        yticklabels: list[str],
        ax: matplotlib.axes.Axes,
        metric: str,
        x_label: str,
        y_label: str,
    ) -> None:
        sns.heatmap(
            data,
            annot = annot, # Display the number in each cell.
            fmt = fmt, # Formats the annotations as percentages with two decimal places.
            annot_kws = annot_kws, # Adjust the font size.
            cmap = cmap, # Use Red-And-Blue colormap.
            xticklabels = xticklabels,
            yticklabels = yticklabels,
            ax = ax,
        )

        ax.set_title(metric)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # Rotate the y-achsis labels to be horizontal.
        ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")

    # This report includes precision, recall and f1-score for each class.
    classification_report_summary = classification_report(
        final_labels_cpu,
        final_predictions_cpu,
        target_names = emotion_labels,
        output_dict = True,
    )
    # Convert the report to DataFrame and exclude the 'accuracy' row.
    report_df = pd.DataFrame(classification_report_summary).transpose()
    # Remove 'accuracy' and support columns.
    x_labels_cr: list[str] = ["precision", "recall", "f1-score"]
    heatmap_df = report_df[x_labels_cr].iloc[:-3]
    # Plot the classification report as heatmap.
    plot_heatmap(
        heatmap_df,
        True,
        ".1%",
        {"size":8},
        "Blues",
        x_labels_cr,
        emotion_labels,
        ax_cr,
        "Classification Report",
        "Variable",
        "Emotion"
    )

    # Create the confusion matrix, to see which emotions are between each are confused by the model.
    confusion_matrix_report = confusion_matrix(
        final_labels_cpu, 
        final_predictions_cpu, 
        normalize="true",
    )
    plot_heatmap(
        confusion_matrix_report,
        True,
        ".1%",
        {"size":8},
        "Reds",
        emotion_labels,
        emotion_labels,
        ax_cm,
        "Normalized Confusion Matrix",
        "Predicted Emotion",
        "True Emotion",
    )

    # Automatically adjust the gaps between graphs, to prevent text overlapping.
    plt.tight_layout(pad=3.0)

#! Unmark these part, whenever you not want to save the model parameters and accuracy graph.

# Create a Timestamped directory for saving

# Get the current date and time.
dir_creation_time = datetime.now()

# Format the timestamp for the directory name: e.g. "12/08/2025_10:30"
new_dir_name: str = dir_creation_time.strftime("%d-%m-%Y_%H-%M")

# Create the new directory path.
MODEL_SAVE_PATH: str = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Facial Emotion Recognition/Runs/"
new_dir_path: str = os.path.join(MODEL_SAVE_PATH, new_dir_name)

try: # Create a new directory.
    os.makedirs(new_dir_path,)
    print(f"\nDirectory {new_dir_name} has been created succesfully.")
except PermissionError:
    print("New directory creation doesn't have the necessary permissions to write to that file.")
except FileExistsError:
    print("A Directory with an exact name exists.")
else:
    # This else-block only runs if the directory was created successfully.

    # Define the file paths within the new directory.
    best_model_parameters_path = os.path.join(new_dir_path, "best_model_parameters.pth")
    last_model_parameters_path = os.path.join(new_dir_path, "last_model_parameters.pth")
    metric_graph_path = os.path.join(new_dir_path, "graphs.png")

    try: # Save the model's best state_dict.
        torch.save(best_model_parameters, best_model_parameters_path)
    except PermissionError:
        print("Model parameter saving doesn't have the necessary permissions to write to that file.")
    try: # Save the model's last state.
        torch.save(model.state_dict(), last_model_parameters_path)
    except PermissionError:
        print("Model parameter saving doesn't have the necessary permissions to write to that file.")

    try: # Save the Matplotlib.graph.
        plt.savefig(metric_graph_path)
        # plt.show has to be here, because it resets the canvas after its closed.
        #!plt.show()
    except PermissionError:
        print("The Graph doesn't have the necessary permissions to write to that file.")

# Get all the layers and parameters from the pipelines for the summary.
def get_pipeline_transformations(pipeline) -> list[str]:
    pipeline_transformations: list[str] = []

    for i, transform in enumerate(pipeline):

        if i == 0:
            pipeline_transformations.append(f"Augmentation Pipeline Transformations: ")

        if isinstance(transform, T.Grayscale):
            num_output_channels = transform.num_output_channels
            pipeline_transformations.append(f" -> Grayscale(num_output_channels={num_output_channels})")
        elif isinstance(transform, T.RandomHorizontalFlip):
            p = transform.p
            pipeline_transformations.append(f" -> RandomHorizontalFlip(p={p})")
        elif isinstance(transform, T.RandomRotation):
            degrees = transform.degrees
            pipeline_transformations.append(f" -> RandomRotation(degrees={degrees})")
        elif isinstance(transform, T.RandomAffine):
            degrees = transform.degrees
            scale = transform.scale
            shear = transform.shear
            pipeline_transformations.append(f" -> RandomAffine(degrees={degrees}, scale={scale}, shear={shear})")
        elif isinstance(transform, T.RandomPerspective):
            distortion_scale = transform.distortion_scale
            p = transform.p
            pipeline_transformations.append(f" -> RandomPerspective(distortion_scale={distortion_scale}, p={p})")
        elif isinstance(transform, T.ColorJitter):
            brightness = transform.brightness
            contrast = transform.contrast
            pipeline_transformations.append(f" -> ColorJitter(brightness={brightness}, contrast={contrast})")
        elif isinstance(transform, T.Resize):
            size = transform.size
            pipeline_transformations.append(f" -> Resize(size={size})")
        elif isinstance(transform, T.RandomErasing):
            p = transform.p
            scale = transform.scale
            ratio = transform.ratio
            value = transform.value
            pipeline_transformations.append(f" -> RandomErasing(p={p}, scale={scale}, ratio={ratio}, value={value})")
        elif isinstance(transform, T.Normalize):
            mean = transform.mean
            std = transform.std
            pipeline_transformations.append(f" -> Normalize(mean={mean}, std={std})")
        elif isinstance(transform, T.ToTensor):
            pipeline_transformations.append(f" -> ToTensor()")

    return pipeline_transformations

try: # Save the report as .txt file to the newly created directory.
    report_path: str = os.path.join(new_dir_path, "summary.txt")

    # Get all the layers for the summary.
    cnn_layers: list[str] = []
    layer_count: int = 1
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            cnn_layers.append(f"Layer {layer_count}:")
            layer_count += 1
        elif isinstance(module, nn.Conv2d):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            padding = module.padding
            cnn_layers.append(f" -> Conv2d(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}, padding={padding})")
        elif isinstance(module, nn.MaxPool2d):
            kernel_size = module.kernel_size
            stride = module.stride
            cnn_layers.append(f" -> MaxPool2d(kernel_size={kernel_size}, stride={stride})")
        elif isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            cnn_layers.append(f" -> Linear(in_features={in_features}, out_features={out_features})")
        elif isinstance(module, nn.Dropout):
            p = module.p
            cnn_layers.append(f" -> Dropout(p={p})")
        elif isinstance(module, nn.Dropout2d):
            p = module.p
            cnn_layers.append(f" -> Dropout2d(p={p})")
        elif isinstance(module, (nn.ReLU)):
            cnn_layers.append(f" -> {type(module).__name__}()")

    # Get all the optimizer parameters for the summary.
    # The optimizer's parameters are stored in a dictionary called "param_groups"
    optimizer_type = type(optimizer).__name__
    optimizer_params = optimizer.param_groups[0]
    learning_rate = optimizer_params["lr"]
    weight_decay = optimizer_params["weight_decay"]

    # We don't set any parameters for loss function.
    loss_function_type = type(criterion).__name__

    # Scheduler parameters are direct attributes of the object.
    scheduler_type = type(scheduler).__name__
    scheduler_mode = scheduler.mode
    scheduler_factor = scheduler.factor
    scheduler_patience = scheduler.patience
    scheduler_min_lrs = scheduler.min_lrs

    standard_pipeline_transformations = get_pipeline_transformations(standard_pipeline.transforms)

    # Get the pipeline type for both training and test.

    with open(report_path, "w") as f:
        f.write(f"\nTotal Training Time: {int(hours)}h {int(minutes)}m {int(seconds)}s, on device: {device}\n")
        f.write(f"\n")
        f.write("\nParameters are:\n")
        f.write(f" - Number of epochs: {num_epochs}\n")
        f.write(f" - Batch size: {BATCH_SIZE}\n")
        f.write(f" - Started with the learning rate {STARTING_LEARNING_RATE} and ended with {new_lr}\n")
        f.write(f" - Best performing model at epoch {best_epoch} with accuracy {best_acc:.2f}% and loss {best_loss:.4f}\n")
        f.write(f"\nComponents:\n")
        f.write(f"  Optimizer:\n")
        f.write(f"  -> Type: {optimizer_type}\n")
        f.write(f"  -> Starting learning rate: {learning_rate}\n")
        f.write(f"  -> Weight decay: {weight_decay}\n")
        f.write(f"  Loss Function:\n")
        f.write(f"  -> Type: {loss_function_type}\n")
        f.write(f"  Scheduler:\n")
        f.write(f"  -> Type: {scheduler_type}\n")
        f.write(f"  -> Mode: {scheduler_mode}\n")
        f.write(f"  -> Factor: {scheduler_factor}\n")
        f.write(f"  -> Patience: {scheduler_patience}\n")
        f.write(f"  -> Minimum learning rate: {scheduler_min_lrs}\n")

        f.write("\nArchitecture: ")
        # Write the layer structure.
        for layer in cnn_layers:
            # Leave space between all the layers.
            if "Layer" in layer:
                f.write("\n")
            f.write(f"\n{layer}")
        f.write("\n")

        for stand_transform in standard_pipeline_transformations:
            f.write(f"\n{stand_transform}")
            if "Pipeline" in stand_transform:
                f.write("\n")
        f.write("\n")
    print("\nReport also saved in the new directory.")
except FileNotFoundError:
    print("File couldn't be found, so the saving of the report was unsuccesful.")

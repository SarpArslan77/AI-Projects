# main.py

from typing import Tuple

from torch.utils.data import DataLoader

from data_preprocess import (
    ConfigDataManager,
    DataManager,
)
from graph import (
    ConfigHistoryPlotter,
    HistoryPlotter
)
from lstm import (
    ConfigLSTM,
    LSTM
)
from train import (
    ConfigTrainer,
    Trainer
)

if __name__ == "__main__":
    # 1. Define the configuration and instance for data preprocessing.
    config_data_manager = ConfigDataManager(
        num_count = 25000,
        sequence_length = 250,
        multiplier = 17,
        increment = 5,
        modulus = 2**8,
        weak_nums_seed = 69, # Arbitrary starting point for the weak numbers.
        stride = 1,
        strong_num_split = 0.5, # Weak number split is calculated upon this.
        train_split = 0.70,
        val_split = 0.20, # Test split is calculated upon this.
        batch_size = 32
    )

    data_manager = DataManager(config_data_manager=config_data_manager)

    # 2. Preprocess the data.
    data_manager.prepare_data()

    # 3. Return the dataloaders.
    dataloaders: Tuple[DataLoader, DataLoader, DataLoader] = data_manager.get_loaders()

    # 4. Define the configuration and instance for LSTM Architecture.
    config_lstm = ConfigLSTM(
        input_size = 1, # Refers to the Feature Dimension (the last number).
        hidden_size = 32,
        num_layers = 1,
        output_size = 1, # Due to binary classification.
        bidirectional = False,
        dropout_p = 0.2

    )

    model = LSTM(config_lstm=config_lstm)

    # 5. Define the configuration and instance for training.
    config_trainer = ConfigTrainer(
        dataloaders = dataloaders,
        learning_rate = 1e-3,
        num_epochs = 200,
        max_norm = 1.0,
        print_freq = 1,
        val_freq = 1,
        save_checkpoint_freq = 20,
        model_save_path = "C:/Users/Besitzer/Desktop/Python/AI Projects/Long-Short Time Memory/Distinguisher/Trials",
        saved_checkpoint = None,
        show_graph = True
    )

    trainer = Trainer(
        config_trainer = config_trainer,
        model = model
    )

    # 6. Start the training.
    acc_history, loss_history, train_time = trainer.fit(test_after_training=True)

    # 7. Define the configuration and instance for graphs.
    config_history_plotter = ConfigHistoryPlotter(
        num_epochs = 200, # TODO Copple it with the num_epochs in ConfigTrainer.
        train_time = train_time
    )

    history_plotter = HistoryPlotter(config_history_plotter=config_history_plotter)

    # 8. Plot the graphs.
    history_plotter.plot_history(
        histories = (acc_history, loss_history),
        trial_folder_path = trainer.trial_folder_path, # The path gets created in the Trainer class.
        show_graph = True
    )



#* main.py

#* ======= Library Imports =======
# ------- Native Library. -------
from datetime import datetime
import sys
from typing import TextIO
import os

# ------- External Libraries. -------
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# Controls whether this file is in a kaggle notebook environment.
is_running_in_kaggle: bool = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

# ------- Custom Libraries. -------
# IF the file is running in kaggle, all other modular files are already in different cells, so no need to import.
if not is_running_in_kaggle:
    from load_data import(
        ConfigDataLoader,
        DataLoader
    )
    from plot_graph import (
        ConfigGraphPlotter,
        GraphPlotter
    )
    from preprocess_data import(
        ConfigDataPreprocessor,
        DataPreprocessor   
    ) 
    from model import (
        ConfigGBDTCatboostClassifier,
        GBDTCatboostClassifier,
        ConfigGBDTCatboostRegressor,
        GBDTCatboostRegressor
    )
    from log_terminal import TerminalLogger

if __name__ == "__main__":
    #* ======= Joint Parameters =======
    #? '__file__': Built-in variable that python automatically creates for every script,
    #   It contains the path to the script currently being executed, stored as string.
    #? .resolve(): Makes the path absolute.
    #   f.e. 'main.py' -> '/Users/username/projects/my_app/main.py'.
    #? .parent: Returns the directory containing the current path.
    #   f.e. '/Users/username/projects/my_app/main.py' -> '/Users/username/projects/my_app'
    # Initializes them based on whether the file is in a local or kaggle environment.
    if is_running_in_kaggle:
        # Sets main path to working directory of kaggle.
        MAIN_PATH = Path("/kaggle/working")

        # Sets the other data and csv files into kaggle dataset paths.
        #! Name of the datasets might be different.
        OTHER_DATA_FILES_PATH: Path = Path("/kaggle/input/datasets/sarparslan07/regional-train-delay-predicter-other-data-files") 
        DATASET_CSV_FILES_PATH: Path = Path("/kaggle/input/datasets/sarparslan07/regional-train-delay-predicter-dataset-csv-files")
    else:
        MAIN_PATH = Path(__file__).resolve().parent

        OTHER_DATA_FILES_PATH: Path = MAIN_PATH / "Other Data Files" 
        DATASET_CSV_FILES_PATH: Path = MAIN_PATH / "Dataset CSV Files" 

    CATEGORICAL_FEATURES: list[str] = [
        "station_name",
        "line_number",
        "final_destination_station",
        "train_type",
        "train_line_station_num",
        "train_number"
    ]

    TRAINING_DATA_FOLDER_NAME: str = "Training Data"
    CURRENT_TRAINING_FOLDER_NAME = Path(datetime.now().strftime(format = "%Y-%m-%d_%H-%M-%S"))
    CURRENT_TRAINING_PATH: Path = MAIN_PATH / TRAINING_DATA_FOLDER_NAME / CURRENT_TRAINING_FOLDER_NAME

    CACHE_FILES_PATH: Path = MAIN_PATH / "Cache Files"

    CATEGORY_LIMITS: list[int] = [5, 15, 30, 60]
    CATEGORY_NAMES: list[str] = [
        "On Time",
        "Minor Delay",
        "Moderate Delay",
        "Severe Delay",
        "Extreme Delay"
    ]

    # ------- Creates the current training directory. -------
    #? parents: Determines whether Python should automatically create any missing parent directories in the path.
    #? exist_ok: Decides how Python reacts if the folder already exists:
    #   True: Ignores it and proceeds without raising an error.
    #   False: Raises a 'FileExistsError' and stops the execution of the script. 
    CURRENT_TRAINING_PATH.mkdir(
        parents = True, 
        exist_ok = True
    )

    # Creates the 'Cache Files' folder so Pandas has a valid folder to write to.
    CACHE_FILES_PATH.mkdir(
        parents = True,
        exist_ok = True
    )

    # ------- Terminal Logger (TL). -------
    terminal_logger = TerminalLogger(file_path = CURRENT_TRAINING_PATH / "terminal_output.txt")

    # Stores the original 'stdout' first so we can restore it later if needed.
    original_stdout: TextIO = sys.stdout
    # Redirect Python's standard output stream.
    sys.stdout = terminal_logger

    #* ======= Module Initializations =======
    print()
    print("=== Starting with Module Initializations... ===")
    # ------- Data Loader (DL). -------
    config_data_loader = ConfigDataLoader(
        should_download_dataset = False,

        selected_german_state = "North Rhine-Westphalia",
        stations_dir = OTHER_DATA_FILES_PATH,
        stations_json_file_name = "all_stations.json",
        date_intervall = (datetime(2024, 7, 1, 0, 0, 0), datetime(2026, 7, 30, 23, 59, 59)),

        dataset_csv_files_path = DATASET_CSV_FILES_PATH,
        chunk_size = 1000,
        to_be_dropped_categorical_features = [
            "id",
            "xml_station_name",
            "eva",
            "time",
            "arrival_change_time",
            "departure_change_time"
        ],
        time_categorical_features = [
            "arrival_planned_time",
            "departure_planned_time"
        ]
    )

    data_loader = DataLoader(
        config_data_loader = config_data_loader
    )

    # ------- Data Preprocesser (DP). -------
    # Since the input directory in kaggle is read-only, metadata must be written into working directory.
    METADATA_FILE_NAME: str = "metadata.json"

    if is_running_in_kaggle:
        METADATA_FILE_PATH: Path = MAIN_PATH / METADATA_FILE_NAME
    else:
        METADATA_FILE_PATH: Path = OTHER_DATA_FILES_PATH / METADATA_FILE_NAME

    config_data_preprocessor = ConfigDataPreprocessor(
        dataset_csv_files_path = DATASET_CSV_FILES_PATH,
        metadata_file_path = METADATA_FILE_PATH,
        cache_file_path = CACHE_FILES_PATH,

        train_val_and_test_dataset_percentage = 0.2,
        train_val_and_test_split_seed = 31,

        validate_and_test_dataset_percentage = 0.5,
        validate_and_test_split_seed = 69,

        categorical_features = CATEGORICAL_FEATURES,

        bin_edges = [
            -np.inf,
            5,
            15,
            30,
            60,
            np.inf
        ],

        cache_keys = [
            "class_X_train", "class_y_train", "class_X_val", "class_y_val", "class_X_test", "class_y_test",
            "reg_X_train", "reg_y_train", "reg_X_val", "reg_y_val", "reg_X_test", "reg_y_test",
            "y_test_reg_unfiltered", "y_test_reg_filtered"
        ]
    )

    data_preprocessor = DataPreprocessor(
        config_data_preprocessor = config_data_preprocessor
    )

    # ------- Gradient Boosted Decision Tree (GBDT) Catboost. -------
    # Classifier.
    config_gbdt_catboost_classifier = ConfigGBDTCatboostClassifier(
        iterations = 3000, 
        learning_rate = 0.02,
        depth = 8,
        verbose = 50,
        max_ctr_complexity = 4,
        boosting_type = "Plain",
        leaf_estimation_method = "Newton",
        catboost_info_target_path = str(CURRENT_TRAINING_PATH),
        auto_class_weights = "Balanced",
        early_stopping_rounds = 100,
        random_seed = 420,
        l2_leaf_reg = 3, 

        categorical_features = CATEGORICAL_FEATURES,
        target_labels = ["Active", "Canceled"],
        model_save_name = "top_catboost_classifier_model.cbm"
    )

    gbdt_catboost_classifier = GBDTCatboostClassifier(config_gbdt_catboost_classifier = config_gbdt_catboost_classifier)

    # Regressor.
    config_gbdt_catboost_regressor = ConfigGBDTCatboostRegressor(
        iterations = 4000, 
        learning_rate = 0.02,
        depth = 8,
        loss_function = "RMSE",
        random_seed = 31,
        verbose = 50,
        catboost_info_target_path = str(CURRENT_TRAINING_PATH),
        early_stopping_rounds = 100,
        l2_leaf_reg = 3, 

        categorical_features = CATEGORICAL_FEATURES,
        category_limits = CATEGORY_LIMITS,
        category_names = CATEGORY_NAMES,
        model_save_name = "top_catboost_regressor_model.cbm"
    )

    gbdt_catboost_regressor = GBDTCatboostRegressor(config_gbdt_catboost_regressor = config_gbdt_catboost_regressor)

    # ------- Graph Plotter (GP). -------
    config_graph_plotter = ConfigGraphPlotter(
        subplots_nrows = 2,
        subplots_ncols = 2,
        subplots_figsize = (16, 16),

        current_training_path = CURRENT_TRAINING_PATH,
        plot_file_name = "result_plots.png",

        cr_annot = True,
        cr_fmt = ".2f",
        cr_annot_kws = {"size": 10},
        cr_cmap = "Blues",
        cr_xticklabels = True,
        cr_yticklabels = True,

        cm_display_labels = ["Canceled"] + CATEGORY_NAMES #TODO FTH
    )

    graph_plotter = GraphPlotter(config_graph_plotter = config_graph_plotter)

    print("=== Finished with Module Initializations. ===")

    # -------- Main Loop -------
    print()
    print("=== Starting with the Main Loop... ===")
    # ------- Data Download / Load. -------
    data_loader.run_dataloader()
        
    # ------- Preprocess the data. -------
    classifier_data, regressor_data, test_regressor_data = data_preprocessor.preprocess_data()

    X_train_filled, y_train_classifier, X_val_filled, y_val_classifier, X_test_filled, y_test_classifier = classifier_data
    X_train_regressor, y_train_regressor_filtered, X_val_regressor, y_val_regressor_filled, X_test_regressor, y_test_regressor_filled = regressor_data
    y_test_regressor, y_test_regressor_filled = test_regressor_data

    # ------- Model Trainings. -------
    # Trains the classifier.
    evals_classifier: dict = gbdt_catboost_classifier.fit(
        train_data = (X_train_filled, y_train_classifier),
        validate_data = (X_val_filled, y_val_classifier)
    )

    # Saves the classifier.
    gbdt_catboost_classifier.save_model()

    # Trains and saves the regressor.
    evals_regressor: dict = gbdt_catboost_regressor.fit(
        train_data = (X_train_regressor, y_train_regressor_filtered),
        validate_data = (X_val_regressor, y_val_regressor_filled)
    )

    gbdt_catboost_regressor.save_model()

    # ------- Evaluates the results on test sets. -------
    # Makes binary predictions on whether the train is canceled.
    classifier_predictions, classifier_prediction_probabilities = gbdt_catboost_classifier.predict(X_test = X_test_filled)

    accuracy, _ = gbdt_catboost_classifier.evaluate(
        y_test = y_test_classifier,
        preds = classifier_predictions
    )

    # Initializes an empty array to hold the final categorical predictions.
    #? np.object_ : An object array stores references to standard Python strings, allowing words of any length to be saved cleanly.
    final_regressor_predictions: NDArray = np.zeros_like(
        a = classifier_predictions,
        dtype = np.object_
    )

    # For the indices where the classifier predicted canceled (= 1), writes 'Canceled' into the final prediction list.
    is_canceled_predicted_indices: NDArray = classifier_predictions == 1

    final_regressor_predictions[is_canceled_predicted_indices] = "Canceled"

    # Extracts the rows from the classifier if the train is predicted to be running (= 0).
    is_predicted_running_indices: NDArray = classifier_predictions == 0

    predicted_running: DataFrame = X_test_filled[is_predicted_running_indices]

    # Evaluates the rows, where it is predicted to be not canceled.
    regressor_predictions_delta: NDArray = gbdt_catboost_regressor.predict(X_test = predicted_running)

    # Reconstructs absolute delays: Absolute = Reference + Delta.
    regressor_predictions_absolute: NDArray =  predicted_running["prev_delay_in_min"].values + regressor_predictions_delta

    # Converts the continuous predictions into categories.
    regressor_predictions_category_indices: NDArray = np.digitize(
        x = regressor_predictions_absolute,
        bins = CATEGORY_LIMITS
    )

    category_names_arr: NDArray = np.array(CATEGORY_NAMES)
    binned_regressor_prediction_categories: NDArray = category_names_arr[regressor_predictions_category_indices]

    # Updates the final predictions.
    final_regressor_predictions[is_predicted_running_indices] = binned_regressor_prediction_categories

    # Initializes an empty array to hold the target labels.
    final_regressor_targets: NDArray = np.zeros_like(
        a = final_regressor_predictions,
        dtype = np.object_
    )

    # Writes the actual cancellations to the target array.
    is_canceled_indices: NDArray = y_test_classifier["is_canceled"].values == True

    final_regressor_targets[is_canceled_indices] = "Canceled"

    # Writes the actual delays.
    is_not_canceled_indices: NDArray = y_test_classifier["is_canceled"].values == False

    not_canceled_rows: DataFrame = y_test_regressor[is_not_canceled_indices]
    not_canceled_rows_flattened: NDArray = not_canceled_rows.values.flatten()

    regressor_targets_category_indices: NDArray = np.digitize(
        x = not_canceled_rows_flattened,
        bins = CATEGORY_LIMITS
    )
    binned_regressor_target_categories: NDArray = category_names_arr[regressor_targets_category_indices]

    final_regressor_targets[is_not_canceled_indices] = binned_regressor_target_categories

    # Gets the regressor evaluation results.

    # ------- Computes the class evaluation metrics. -------
    # Adds the 'Canceled' label to the labels of the regressor model.
    new_category_names: list[str] = ["Canceled"] + CATEGORY_NAMES

    # Creates an classification report (CR).
    cr_dict: dict = classification_report(
        final_regressor_targets,
        final_regressor_predictions,
        labels = new_category_names,
        output_dict = True
    )

    # Creates a confusion matrix (CM).
    #? normalize: Normalizes the CM values within [0, 1] range.
    cm: NDArray = confusion_matrix(
        final_regressor_targets,
        final_regressor_predictions,
        labels = new_category_names,
        normalize = "true"
    )

    # ------- Plots the metrics. -------
    graph_plotter.run_graph_plotter(
        evals_classifier = evals_classifier,
        evals_regressor = evals_regressor,
        classification_report_dict = cr_dict,
        confusion_matrix = cm,
        display_plots = True
    )

    print("=== Finished with the Main Loop! ===")

    # ------- Cleans up the TL. -------
    # Restores the original terminal stream.
    sys.stdout = original_stdout
    terminal_logger.close()

#* model.py

#* ======= Library Imports =======
# ------- Native Libraries. -------
from dataclasses import dataclass
import os

# ------- Export Libraries. -------
from catboost import (
    CatBoostClassifier,
    CatBoostRegressor,
    Pool
)

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,

    mean_absolute_error,
    root_mean_squared_error,
    r2_score,

    classification_report,
    confusion_matrix
)

#* ======= Classifier Data Class Initialization =======
@dataclass(frozen=True)
class ConfigGBDTCatboostClassifier:
    """
    Configuration schema for the Gradient Boosted Decision Tree (GBDT) model.

    This dataclass encapsulates all hyperparameters, algorithmic settings, and 
    metadata required to initialize and train a CatBoost classifier.

    Attributes:
        iterations (int): Maximum number of trees to be built during training.
        learning_rate (float): Step size shrinkage used during update to prevent overfitting.
        depth (int): Depth of the decision trees.
        verbose (int): Interval for logging training metrics to the console.
        max_ctr_complexity (int): Maximum number of features that can be combined 
            for categorical target representation.
        boosting_type (str): The boosting scheme to be used (e.g., 'Plain', 'Ordered').
        leaf_estimation_method (str): Algorithm used to calculate numerical leaf values.
        catboost_info_target_dir (str): Directory path for training logs and metadata.
        auto_class_weights (str): Automatically re-weight the classes so that minority 
            classes are penalized heavily if the model gets them wrong.
        early_stopping_rounds (int): Limit for consecutive iterations without 
            metric improvement before training terminates.
            
        categorical_features (list[str]): Column names identifying categorical data.
        target_labels (list[str]): Descriptive names for the output classification bins.
        model_save_name (str): Filename used for exporting the trained model to disk.
    """
    # ------- Model Training Parameters. -------
    iterations: int
    learning_rate: float
    depth: int
    verbose: int
    max_ctr_complexity: int
    boosting_type: str
    leaf_estimation_method: str
    catboost_info_target_path: str
    auto_class_weights: str
    early_stopping_rounds: int
    random_seed: int
    l2_leaf_reg: int

    # ------- Data Parameters. -------
    categorical_features: list[str]
    target_labels: list[str]
    model_save_name: str

#* ======= Gradient Boosted Decision Tree (GBDT) Catboost Class Initialization =======
class GBDTCatboostClassifier:
    def __init__(
            self,
            config_gbdt_catboost_classifier: ConfigGBDTCatboostClassifier
    ) -> None:
        # ------- Initializes class attributes from the parameters. -------
        self.classifier_cfg: ConfigGBDTCatboostClassifier = config_gbdt_catboost_classifier

        # ------- Initializes class attributes dynamically. -------
        #? use_best_model: Ensures the final model uses the best iterations trees.
        #? thread_count = -1: Utilizes all cores on the processor.
        #? border_count: Dictates how many splits are evaluated for numerical features.
        #? rsm (Random Subspace Method): Controls % of features selected at random to evaluate for each split.
        self.classifier_model = CatBoostClassifier(
            iterations = self.classifier_cfg.iterations,
            learning_rate = self.classifier_cfg.learning_rate,
            depth = self.classifier_cfg.depth,
            verbose = self.classifier_cfg.verbose,
            random_seed = self.classifier_cfg.random_seed,
            l2_leaf_reg = self.classifier_cfg.l2_leaf_reg,
            
            max_ctr_complexity = self.classifier_cfg.max_ctr_complexity,
            boosting_type = self.classifier_cfg.boosting_type,
            leaf_estimation_method = self.classifier_cfg.leaf_estimation_method,

            train_dir = self.classifier_cfg.catboost_info_target_path,

            early_stopping_rounds = self.classifier_cfg.early_stopping_rounds,
            auto_class_weights = self.classifier_cfg.auto_class_weights,
            #!task_type = "GPU",
            use_best_model = True,
            thread_count = -1,
            border_count = 128,
        )

        print("-> GBDT Catboost Classifier initialized! ")

    def fit(
            self,
            train_data: tuple[DataFrame, DataFrame], # (X_train, y_train).
            validate_data: tuple[DataFrame, DataFrame]
    ) -> dict:
        print()
        print("--- Starting with training the classifier... ---")

        # ------- Unpacks the parameters. -------
        X_train, y_train = train_data
        X_val, y_val = validate_data

        # ------- Optimizes the datasets for speed. -------
        # Wraps the datasets in CatBoost Pools.
        train_pool = Pool(
            data = X_train,
            label = y_train,
            cat_features = self.classifier_cfg.categorical_features
        )
        val_pool = Pool(
            data = X_val,
            label = y_val,
            cat_features = self.classifier_cfg.categorical_features
        )

        # Pre-quantizes features.
        train_pool.quantize()

        # ------- Fits the model with training data. -------
        self.classifier_model.fit(
            X = train_pool,
            eval_set = val_pool
        )

        print("--- Finished with training the classifier. ---")

        return self.classifier_model.get_evals_result()

    def predict(
            self,
            X_test: DataFrame
    ) -> tuple[NDArray, NDArray]:
        print(f"--> Performing classifier inference on test samples...")

        # ------- Predicts the output based on test data. -------
        preds: NDArray = self.classifier_model.predict(data = X_test)

        # ------- Calculates the prediction probabilities -------
        pred_probs: NDArray = self.classifier_model.predict_proba(X = X_test)

        print("--> Predicted the result and calculated the probabilities.")

        return (preds, pred_probs)
    
    def evaluate(
            self,
            y_test: DataFrame,
            preds: NDArray,
    ) -> tuple[float, str | dict[str, any]]:
        print(f"---> Evaluating classifier on instances...")

        # ------- Calculates the overall accuracy. -------
        acc: float = accuracy_score(
            y_test,
            preds
        )

        # ------- Identifies which target classes are present in the current test batch. -------
        predictions_flattened: NDArray = preds.flatten() 
        y_test_flattened: NDArray = y_test.values.flatten()

        present_target_class_indices: list[int] = np.unique(
            np.concatenate([y_test_flattened, predictions_flattened])
        ).astype(int)

        # Filters the 'TARGET_LABELS' so the size matches the data.
        present_target_labels: list[str] = [
            self.classifier_cfg.target_labels[i] for i in present_target_class_indices
        ]

        # ------- Creates the classification report. -------
        class_rep: str | dict[str, any] = classification_report(
            y_test,
            preds,
            target_names = present_target_labels
        )

        print("---> Evaluated the results and created a classification report.")
        print("Classification Report: ")
        print(class_rep)

        return (acc, class_rep)
    
    def save_model(self) -> None:
        # ------- Saves the model. -------
        self.classifier_model.save_model(fname = (Path(self.classifier_cfg.catboost_info_target_path) / self.classifier_cfg.model_save_name))

        print("-> Succesfully saved the model.")

#* ======= Regressor Data Class Initialization =======
@dataclass(frozen = True)
class ConfigGBDTCatboostRegressor:
    #TODO AD
    # ------- Model Training Parameters. ---
    iterations: int
    learning_rate: float
    depth: int
    loss_function: str
    random_seed: int
    verbose: int
    catboost_info_target_path: str
    early_stopping_rounds: int
    l2_leaf_reg: int

    # ------- Data Parameters. -------
    categorical_features: list[str]
    category_limits: list[int]
    category_names: list[str]
    model_save_name: str

#* ======= GBDT Catboost Regressor Class Initialization =======
class GBDTCatboostRegressor:
    def __init__(
            self,
            config_gbdt_catboost_regressor: ConfigGBDTCatboostRegressor
    ) -> None:
        # ------- Initializes class attributes from parameters. -------
        self.regressor_cfg: ConfigGBDTCatboostRegressor = config_gbdt_catboost_regressor

        # ------- Initializes class attributes dynamically. -------
        self.regressor_model = CatBoostRegressor(
            iterations = self.regressor_cfg.iterations,
            learning_rate = self.regressor_cfg.learning_rate,
            depth = self.regressor_cfg.depth,
            loss_function = self.regressor_cfg.loss_function,
            random_seed = self.regressor_cfg.random_seed,
            verbose = self.regressor_cfg.verbose,
            train_dir = self.regressor_cfg.catboost_info_target_path,
            early_stopping_rounds = self.regressor_cfg.early_stopping_rounds,
            use_best_model = True,
            l2_leaf_reg = self.regressor_cfg.l2_leaf_reg,
            boosting_type = "Plain",
            thread_count = os.cpu_count() // 2,
            border_count = 64,
            #!task_type = "GPU"
        )

        print("-> GBDT Catboost Regressor initialized! ")

    def fit(
            self,
            train_data: tuple[DataFrame, DataFrame], # (X_train, y_train).
            validate_data: tuple[DataFrame, DataFrame] # (X_validate, y_validate).
    ) -> dict:
        print()
        print("--- Starting with training the regressor... ---")

        # ------- Unpacks the parameters. -------
        X_train, y_train = train_data
        X_val, y_val = validate_data

        # ------- Optimizes the datasets for speed. -------
        # Wraps the datasets in CatBoost Pools.
        train_pool = Pool(
            data = X_train,
            label = y_train,
            cat_features = self.regressor_cfg.categorical_features
        )
        val_pool = Pool(
            data = X_val,
            label = y_val,
            cat_features = self.regressor_cfg.categorical_features
        )

        # Pre-quantizes the features.
        train_pool.quantize()

        # ------- Fits the model with training data. -------
        self.regressor_model.fit(
            X = train_pool,
            eval_set = val_pool
        )

        print("--- Finished with training the regressor. ---")

        return self.regressor_model.get_evals_result()

    def predict(
            self,
            X_test: DataFrame
    ) -> NDArray:
        print(f"--> Performing regressor inference on test samples...")

        # ------- Predicts the output based on test data. -------
        preds: NDArray = self.regressor_model.predict(data = X_test)

        print("--> Predicted the results.")

        return preds
    
    def evaluate(
            self,
            y_test: DataFrame,
            preds: NDArray
    ) -> tuple[dict, NDArray]:
        print(f"---> Evaluating regressor on instances...")
        
        # ------- Evaluates the predictions in continuous space for regression metrics. -------
        # Mean Absolute Error (MAE).
        mae: NDArray = mean_absolute_error(
            y_test,
            preds
        )

        # Root Mean Squared Error (RMSE).
        rmse: NDArray = root_mean_squared_error(
            y_test,
            preds
        )

        # R-squared (R^2).
        r2: NDArray = r2_score(
            y_test,
            preds
        )

        print("---> Regression Results:")
        print(f"    MAE: {mae:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    R^2: {r2:.4f}")

        # ------- Converts the regression metrics into classification metrics. -------
        # Applies the interval boundaries. 
        predicted_category_indices: NDArray = np.digitize(
            x = preds,
            bins = self.regressor_cfg.category_limits
        )

        # Applies the category names.
        category_names_arr: NDArray = np.array(self.regressor_cfg.category_names)
        binned_prediction_categories: NDArray = category_names_arr[predicted_category_indices]

        # Repeats the same process for the truth labels.
        y_test_arr: NDArray = np.array(y_test)
        y_test_flattened: NDArray = y_test_arr.flatten()

        actual_category_indices: NDArray = np.digitize(
            x = y_test_flattened,
            bins = self.regressor_cfg.category_limits
        )

        binned_y_test: NDArray = category_names_arr[actual_category_indices]

        # ------- Evaluates the predictions in binned space for classification metrics. -------
        # Creates a classification report.
        report_dict: str = classification_report(
            binned_y_test,
            binned_prediction_categories,
            labels = self.regressor_cfg.category_names,
            output_dict = True #TODO ET
        )

        # Creates a confusion matrix and displays it.
        cm: NDArray = confusion_matrix(
            binned_y_test,
            binned_prediction_categories,
            labels = self.regressor_cfg.category_names,
            normalize = "true"
        )

        print("----> Evaluated the results and binned to category names!")

        return (report_dict, cm)
    
    def save_model(self) -> None:
        # ------- Saves the model. -------
        model_save_dir: Path = Path(self.regressor_cfg.catboost_info_target_path) / self.regressor_cfg.model_save_name

        self.regressor_model.save_model(fname = model_save_dir)

        print()
        print("-> Succesfully saved the model.")


#* preprocess_data.py

#TODO:
#   FTH: Fix the hardcoding.

#* ======= Library Imports =======
# ------- Native Libraries. -------
from dataclasses import (
    dataclass,
    asdict
)
import json
from os import stat_result
from typing import Any

# ------- External Libraries. -------
import glob
import numpy as np
import pandas as pd
from pandas import (
    DataFrame,
    Series
)
from pathlib import Path
import polars as pl
from sklearn.model_selection import train_test_split

#* ======= Config Data Preprocessor Class Initialization =======
@dataclass(frozen = True)
class ConfigDataPreprocessor:
    dataset_csv_files_path: Path
    metadata_file_path: Path
    cache_file_path: Path

    train_val_and_test_dataset_percentage: float
    train_val_and_test_split_seed: int

    validate_and_test_dataset_percentage: float
    validate_and_test_split_seed: int

    categorical_features: list[str]

    bin_edges: list[int]

    cache_keys: list[str]

#* ======= Data Preprocessor Class Initialization =======
# Custom Type Hints.
model_dataset_type = tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]
preprocess_return_type = tuple[model_dataset_type, model_dataset_type, tuple[DataFrame, DataFrame]]

class DataPreprocessor:
    def __init__(
            self,
            config_data_preprocessor: ConfigDataPreprocessor
    ) -> None:
        self.cfg: ConfigDataPreprocessor = config_data_preprocessor

        print("-> Data Preprocesser Class initialized! ")
    
    def _combine_datasets(
            self,
            classifier_data: model_dataset_type,
            regressor_data: model_dataset_type,
            extra_data: tuple[DataFrame, DataFrame]
    ) -> dict[str, DataFrame]:
        # Converts all the dataframes into lists.
        classifier_data_list: list = list(classifier_data)
        regressor_data_list: list = list(regressor_data)
        extra_data_list: list = list(extra_data)

        # Combines the lists.
        combined_dataset_list: list[DataFrame] = classifier_data_list + regressor_data_list + extra_data_list

        # Converts the combines dataset list into a dictionary.
        combined_dataset_dict: dict[str, DataFrame] = dict(
            zip(self.cfg.cache_keys, combined_dataset_list)
        )

        return combined_dataset_dict
    
    def _unpack_datasets(
            self,
            combined_dataset_dict: dict[str, DataFrame]
    ) -> preprocess_return_type:
        # Reconstructs the original 14-DataFrame nested tuple from the combined dictionary of DataFrames.
        classifier_data: model_dataset_type = (
            combined_dataset_dict["class_X_train"],
            combined_dataset_dict["class_y_train"],
            combined_dataset_dict["class_X_val"],
            combined_dataset_dict["class_y_val"],
            combined_dataset_dict["class_X_test"],
            combined_dataset_dict["class_y_test"]
        )

        regressor_data: model_dataset_type = (
            combined_dataset_dict["reg_X_train"],
            combined_dataset_dict["reg_y_train"],
            combined_dataset_dict["reg_X_val"],
            combined_dataset_dict["reg_y_val"],
            combined_dataset_dict["reg_X_test"],
            combined_dataset_dict["reg_y_test"]
        )

        extra_data: tuple[DataFrame, DataFrame] = (
            combined_dataset_dict["y_test_reg_unfiltered"],
            combined_dataset_dict["y_test_reg_filtered"]
        )

        return (classifier_data, regressor_data, extra_data)

    def _get_metadata_files(
            self
    ) -> dict:
        metadata: dict = {}

        # Ensures files are sorted.
        for dataset_csv_file_path in sorted(self.cfg.dataset_csv_files_path.glob("*.csv")):
            #? .stat():  Retrieves physical information and metadata about the file or directory.
            status_result: stat_result = dataset_csv_file_path.stat()
            #? .st_mtime: Modification Time, represents the timestamp of when the file's contents were last modified to.
            #? .st_size: Physical size of the file, measured in bytes.
            metadata[dataset_csv_file_path.name] = {
                "mtime": status_result.st_mtime,
                "size": status_result.st_size
            }
        
        return metadata
    
    def _is_metadata_same(
            self,
            current_metadata: dict
    ) -> bool:
        # IF cache files are missing, it is automatically invalid.
        if not self.cfg.cache_file_path.exists() or not self.cfg.metadata_file_path.exists():
            return False
        
        # Checks IF all 14 individual parquet files exist in cache directory.
        for key in self.cfg.cache_keys:
            cache_parquet_file_name: str = f"{key}.parquet"
            cache_parquet_file_path: Path = self.cfg.cache_file_path / cache_parquet_file_name

            if not (cache_parquet_file_path.exists()):
                return False
        
        # ------- Loads metadata from the last succesful preprocessing run. -------
        try:
            with open(
                file = self.cfg.metadata_file_path,
                mode = "r",
                encoding = "utf-8"
            ) as f:
                saved_metadata: dict[str, Any] = json.load(f)

            # Checks IF 'csv' files have changed.
            #? .get(): Attempts to retrieve the value associated with a key in a dictionary.
            #   IF it does not exist, instead of crashing, it returns a fallback value. (default = None)
            raw_files: dict[str, dict[str, float | int]] | None = saved_metadata.get("raw_files")
            if raw_files != current_metadata:
                return False 
            
            # Checks if configuration settings have changed.
            config: dict[str, Any] | None = saved_metadata.get("config")

            # Converts any Path objects to strings so they match JSON-deserialized strings.
            #? asdict(): Converts dataclass instance into dictionary.
            cfg_dict: dict = asdict(self.cfg)
            cfg_dict_serializable = self._serialize_config(cfg_dict = cfg_dict)

            if config != cfg_dict_serializable:
                return False
            
            return True
        
        except Exception as e:
            return False
    
    def __to_json(
            self,
            val: Any
    ) -> Any:
        if isinstance(val, Path):
            return str(val)
        
        # Checks native Python float and all NumPy/Pandas floating type variations.
        #? np.floating: Abstract base class in NumPy for all its
        #   float variations. (float16, float32, longdouble etc.)
        if isinstance(val, (float, np.floating)):
            # Handles infinity to prevent raw math literals in JSON files.
            if (val == float("inf")) or (val == np.inf):
                return "inf"
            elif (val == float("-inf") or (val == -np.inf)):
                return "-inf"
            # Normalizes all NumPy/Pandas floats to Python floats.
            return float(val)
        
        # Recursively evaluates elements inside lists or tuples.
        if isinstance(val, (list, tuple)):
            return [self.__to_json(val = item) for item in val]
        if isinstance(val, dict):
            return {key: self.__to_json(val = v) for key, v in val.items()}
        
        return val

    def _serialize_config(
            self,
            cfg_dict: dict
    ) -> dict:
        return {key: self.__to_json(val = val) for key, val in cfg_dict.items()}

    def _save_metadata(
            self,
            classifier_data: model_dataset_type,
            regressor_data: model_dataset_type,
            extra_data: tuple[DataFrame, DataFrame],
            current_metadata: dict
    ) -> None:
        combined_dataset_dict: dict[str, DataFrame] = self._combine_datasets(
            classifier_data = classifier_data,
            regressor_data = regressor_data,
            extra_data = extra_data
        )

        # Saves each of the 14 outputs to individual Parquet files.
        for key, df in combined_dataset_dict.items():
            parquet_file_name: str = f"{key}.parquet"
            parquet_file_path: Path = self.cfg.cache_file_path / parquet_file_name

            if isinstance(df, Series):
                df: DataFrame = df.to_frame()

            #? index = False: Instructs Pandas not to save the index column.
            df.to_parquet(
                parquet_file_path,
                index = False
            )

        # Packages and saves the validation metadata payloads.
        cfg_dict: dict = asdict(self.cfg)

        # Converts Path objects into strings before saving to JSON.
        cfg_dict_serializable = self._serialize_config(cfg_dict = cfg_dict)

        payload: dict[str, dict] = {
            "raw_files": current_metadata,
            "config": cfg_dict_serializable
        }

        with open(
            file = self.cfg.metadata_file_path,
            mode = "w",
            encoding = "utf-8"
        ) as f:
            #? indent: Instructs the writer to format the JSON file within line breaks and
            #   indent each nested level of data with exactly 4 spaces.
            json.dump(
                payload,
                f,
                indent = 4
            )

    def _split_dataset(
            self,
            X: DataFrame,
            y: DataFrame,
            second_dataframe_size: float,
            random_state: int
    ) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        # Splits the data into two other datasets based on a percentage.
        X_first, X_second, y_first, y_second = train_test_split(
            X,
            y,
            test_size = second_dataframe_size,
            random_state = random_state
        )

        return (X_first, X_second, y_first, y_second)

    def _fill_missing_values(
            self,
            X: DataFrame
    ) -> DataFrame:
        # CatBoost does not allow numerical NaN (float) values in columns designated as categorical.
        #   In pandas, missing values are represented by NaN, which is a float. CatBoost requires all
        #   values in a categorical column to be either integers, strings or bytes. When it encountered
        #   a float nan in a categorical column, it raises an error.
        # ---> Fills the NaNs with a placeholder 'missing' and convert the entire column to string type.
        X_filled: DataFrame = X.copy()

        X_filled[self.cfg.categorical_features] = (
            X[self.cfg.categorical_features].fillna("missing").astype(str)
        )

        return X_filled

    def _filter_canceled_rows(
            self,
            X: DataFrame,
            y: DataFrame,
            is_not_canceled_indices: Series
    ) -> tuple[DataFrame, DataFrame]:
        X_filtered: DataFrame = X[is_not_canceled_indices]

        y_filtered: DataFrame = y[is_not_canceled_indices]

        return (X_filtered, y_filtered)

    def preprocess_data(
            self
    ) -> tuple[
        model_dataset_type, 
        model_dataset_type, 
        tuple[DataFrame, DataFrame]
    ]:
        print()
        print("--- Starting with Preprocessing Data... ---")

        # ------- Controls whether the metadata is same, so the preprocessing can be skipped. -------
        # Gets the metadata for the current files.
        current_metadata: dict = self._get_metadata_files()

        # Checks if preprocessing can be skipped.
        is_metadata_same: bool = self._is_metadata_same(current_metadata = current_metadata)
        if is_metadata_same:
            print("-> Cache valid! Loading preprocessed data instantly...")

            try:
                combined_dataset_dict: dict[str, DataFrame] = {} 

                for key in self.cfg.cache_keys:
                    parquet_file_name: str = f"{key}.parquet"
                    parquet_file_path: Path = self.cfg.cache_file_path / parquet_file_name

                    parquet_file: DataFrame = pd.read_parquet(path = parquet_file_path)

                    combined_dataset_dict[key] = parquet_file

                unpacked_datasets: preprocess_return_type = self._unpack_datasets(
                    combined_dataset_dict = combined_dataset_dict
                )

                print("--- Finished with Preprocessing Data. ---")

                return unpacked_datasets

            except Exception as e:
                print(f"-> Failed to load cached files: {e}. Re-running preprocessing...")

        # ------- Loads the .csv files into a dataframe. -------
        csv_glob = str(self.cfg.dataset_csv_files_path / "*.csv")

        # Reads and concatenates in parallel, then convers to Pandas for CatBoost.
        df: DataFrame = pl.read_csv(csv_glob).to_pandas()

        """
        #DEBUGGING
        df = df.sample(
            frac = 0.1,
            random_state = 31
        ).reset_index(drop = True)
        """

        # ------- Seperates the features (X) and target (y), splits the data into training and validation/test sets. -------
        X: DataFrame = df.drop(
            columns = [
                "delay_in_min",
                "is_canceled"
            ]
        )

        #? [[ ]]: Passing a list of column mames inside double brackets returns a single-column DataFrame.
        y_classifier: DataFrame = df[["is_canceled"]]
        y_regressor: DataFrame = df[["delay_in_min"]]

        # Combines the both target columns, so at the time of split all data stays aligned.
        #? ignore_index: While concatenating horizontally, this parameters removes the column names, IF set to 'True'.
        #? axis: By default (axis = 0) pd.concat() stacks the columns vertically, so axis = 1 stacks them horizontally.
        y_combined: DataFrame = pd.concat(
            [
                y_classifier,
                y_regressor
            ],
            ignore_index = False,
            axis = 1
        )

        # Splits the data into training, validation and test datasets.
        X_train, X_val_and_test, y_train, y_val_and_test = self._split_dataset(
            X = X,
            y = y_combined,
            second_dataframe_size = self.cfg.train_val_and_test_dataset_percentage,
            random_state = self.cfg.train_val_and_test_split_seed
        )

        X_val, X_test, y_val, y_test = self._split_dataset(
            X = X_val_and_test,
            y = y_val_and_test,
            second_dataframe_size = self.cfg.validate_and_test_dataset_percentage,
            random_state = self.cfg.validate_and_test_split_seed
        )

        # Fills the missing values.
        X_train_filled: DataFrame = self._fill_missing_values(X = X_train)
        X_val_filled: DataFrame = self._fill_missing_values(X = X_val)
        X_test_filled: DataFrame = self._fill_missing_values(X = X_test)

        # Seperates the target datasets to classifier and regressor.
        y_train_classifier: DataFrame = y_train[["is_canceled"]]
        y_val_classifier: DataFrame = y_val[["is_canceled"]]
        y_test_classifier: DataFrame = y_test[["is_canceled"]]

        y_train_regressor: DataFrame = y_train[["delay_in_min"]]
        y_val_regressor: DataFrame = y_val[["delay_in_min"]]
        y_test_regressor: DataFrame = y_test[["delay_in_min"]]

        # Filters the regressor dataset so, that only non-canceled rows are present.
        is_not_canceled_indices_train_regressor: DataFrame = (y_train_classifier["is_canceled"] == False) & (y_train_regressor["delay_in_min"].notna())
        X_train_regressor, y_train_regressor_filtered = self._filter_canceled_rows(
            X = X_train_filled,
            y = y_train_regressor,
            is_not_canceled_indices = is_not_canceled_indices_train_regressor
        )

        is_not_canceled_indices_val_regressor: DataFrame = (y_val_classifier["is_canceled"] == False) & (y_val_regressor["delay_in_min"].notna())
        X_val_regressor, y_val_regressor_filtered = self._filter_canceled_rows(
            X = X_val_filled,
            y = y_val_regressor,
            is_not_canceled_indices = is_not_canceled_indices_val_regressor
        )

        is_not_canceled_indices_test_regressor: DataFrame = (y_test_classifier["is_canceled"] == False) & (y_test_regressor["delay_in_min"].notna())
        X_test_regressor, y_test_regressor_filtered = self._filter_canceled_rows(
            X = X_test_filled,
            y = y_test_regressor,
            is_not_canceled_indices = is_not_canceled_indices_test_regressor
        )

        # Organizes the datasets.
        classifier_dataset: model_dataset_type = (
            X_train_filled, y_train_classifier, X_val_filled, y_val_classifier, X_test_filled, y_test_classifier
        )
        regressor_dataset: model_dataset_type = (
            X_train_regressor, y_train_regressor_filtered, X_val_regressor, y_val_regressor_filtered, X_test_regressor, y_test_regressor_filtered
        )
        extra_dataset: tuple[DataFrame, DataFrame] = (
            y_test_regressor, y_test_regressor_filtered
        )

        # ------- Saves outputs as parquet file. -------
        self._save_metadata(
            classifier_data = classifier_dataset,
            regressor_data = regressor_dataset,
            extra_data = extra_dataset,
            current_metadata = current_metadata
        )

        print("--- Finished with Preprocessing Data. ---")

        return (classifier_dataset, regressor_dataset, extra_dataset)
        

#* load_data.py

#* ======= Library Imports =======
# ------- Internal Libraries. -------
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from datasets import (
    load_dataset,
    IterableDataset
)
import json
import sys
from typing import Any

# ------- External Libraries. -------
import numpy as np
import pandas as pd
from pandas import (
    DataFrame,
    Series,
    Timestamp
)
from pathlib import Path

@dataclass(frozen = True)
class ConfigDataLoader:
    """
    Configuration schema for the DataLoader class.

    This dataclass encapsulates dataset sourcing options (online vs. local paths),
    geographic and temporal filtering criteria for specific German weather stations,
    and parameters governing preprocessing and chunk-based CSV exportation.

    Attributes:
        should_download_dataset (bool): Flag determining whether to stream the dataset from Hugging Face or load it locally.
        
        selected_german_state (str): The specific German state name used to filter target weather stations.
        stations_json_file_name (str): The filename of the JSON document containing state-to-station mappings.
        stations_dir (Path): The directory path containing the station mappings JSON file.
        date_interval (tuple[datetime, datetime]): A datetime range (start, end) used to filter records chronologically.
        
        dataset_csv_files_path (Path): The target directory where exported and chunked CSV files are saved or read.
        chunk_size (int): The maximum buffer limit per year-month grouping before writing rows to disk.
        to_be_dropped_categorical_features (list[str]): Columns to discard during the export/preprocessing stage.
        time_categorical_features (list[str]): Timestamp columns to be transformed into cyclical sine and cosine features.
    """

    # ------- Dataset Source Configuration. -------
    should_download_dataset: bool

    # ------- Filtering & Geographic Configuration. -------
    selected_german_state: str 
    stations_json_file_name: str
    stations_dir: Path  
    date_interval: tuple[datetime, datetime]

    # ------- Export & Processing Configuration. -------
    dataset_csv_files_path: Path
    chunk_size: int
    to_be_dropped_categorical_features: list[str]
    time_categorical_features: list[str]

#* ======= Data Loader Class Initialization =======
class DataLoader:
    def __init__(
            self,
            config_data_loader: ConfigDataLoader
    ) -> None:
        # ------- Initializes class attributes from parameters. -------
        self.cfg: ConfigDataLoader = config_data_loader

        # ------- Initializes class attributes dynamically. -------
        self.dataset: IterableDataset | None = None

        print("-> Data Loader Class initialized! ")

    def _download_online_dataset(self) -> None:
        # ------- Downloads the dataset in streaming mode from HuggingFace. -------
        self.dataset = load_dataset(
            path = "piebro/deutsche-bahn-data",
            data_files = "monthly_processed_data/*.parquet",
            streaming = True
        )

    def _load_local_dataset(self) -> None:
        # ------- Loads the dataset locally from a directory path. -------
        csv_file_dirs_str = str(self.cfg.dataset_csv_files_path / "*.csv")

        self.dataset = load_dataset(
            path = "csv", 
            data_files = csv_file_dirs_str,
            streaming = True
        )

    def __filter_date(
            self,
            data : dict[str, Any],
            date_intervall: tuple[datetime, datetime]
    ) -> bool:
        # ------- Unpacks the parameters. -------
        start_date, end_date = date_intervall

        # ------- Strips the date of the data and checks whether is it in the date interval. -------
        data_date: Timestamp = data["time"]
        localized_data_date: Timestamp = pd.to_datetime(data_date).tz_localize(None)

        is_in_interval: bool = start_date <= localized_data_date <= end_date

        return is_in_interval

    def _filter_dataset(self) -> None:
        # ------- Loads the station names of selected state from the .json file. -------
        stations_json_file_dir: Path = self.cfg.stations_dir / self.cfg.stations_json_file_name
        
        #? 'utf-8': Unicode Transformation Format- 8-bit.
        #   Ensures that python interprets the bytes correctly, especially if the file contains non-English letters.
        with open(
            file = stations_json_file_dir,
            mode = "r",
            encoding = "utf-8"
        ) as file:
            all_stations: dict[str, str] = json.load(file)

        # ------- Unpacks the selected state. -------
        selected_stations: list[str] = all_stations[self.cfg.selected_german_state]

        # ------- Filters the stream-dataset on the fly. -------
        self.dataset: IterableDataset = self.dataset.filter(
            lambda x:
                x["station_name"] in selected_stations and
                self.__filter_date(
                    data = x,
                    date_intervall = self.cfg.date_interval
                )
        )

        self.dataset = self.dataset["train"] #TODO FTH

    def __process_and_save_chunk(
            self,
            chunk: list[dict],
            year: int,
            month: int
    ) -> None:
        # If there is not data, return immediately.
        if not chunk:
            return

        df = DataFrame(chunk)

        # Removes unnecessary categorical features.
        dropped_df: DataFrame = df.drop(columns = self.cfg.to_be_dropped_categorical_features)

        # Converts timestamp categories into cyclical numerical values.
        for time_category in self.cfg.time_categorical_features:
            dt_series: Series[Timestamp] = pd.to_datetime(arg = df[time_category])

            # Maps the time units to their series and periods.
            cycles: dict[str, tuple[Series, int]] = {
                "second": (dt_series.dt.second, 60),
                "minute": (dt_series.dt.minute, 60),
                "hour": (dt_series.dt.hour, 24),
                "day": (dt_series.dt.day, dt_series.dt.days_in_month),
                "day_of_week": (dt_series.dt.day_of_week, 7),
                "month": (dt_series.dt.month, 12),
            }

            # Calculates dynamically 'sin' and 'cos' for each time unit.
            for unit, (value, period) in cycles.items():
                dropped_df[f"{time_category}_{unit}_sin"] = np.round(
                    a = np.sin(2 * np.pi * (value / period)),
                    decimals = 4
                )
                dropped_df[f"{time_category}_{unit}_cos"] = np.round(
                    a = np.cos(2 * np.pi * (value / period)),
                    decimals = 4
                )

            dropped_df[f"{time_category}_year"] = dt_series.dt.year

            # Deletes the old datetime category.
            final_dropped_df: DataFrame = dropped_df.drop(columns = time_category)

        # Constructs target file path for each csv file.
        csv_file_name: str = f"{year}_{month:02d}.csv"
        csv_file_path: Path = self.cfg.dataset_csv_files_path / csv_file_name

        # Writes the header, IF the file is being created for the very first time.
        should_write_header: bool = not csv_file_path.exists()

        final_dropped_df.to_csv(
            path_or_buf = csv_file_path,
            mode = "a",
            index = False,
            header = should_write_header
        )
    
    def _export_dataset_as_csv(self) -> None:
        # ------- Asks to delete old files, IF they exist already. -------
        csv_files = list(self.cfg.dataset_csv_files_path.glob("*.csv"))

        if csv_files:
            valid_answer: bool = False

            while (not valid_answer):
                print()
                print("!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!")
                #? .strip(): Removes extra spaces and specific chars from the start and end of a text string.
                user_input: str = input(
                    f"Old CSV files found in {self.cfg.dataset_csv_files_path}\n"
                    "Do you want to delete them and proceed with the new download? (Y/n): "
                ).strip()

                if user_input == "Y":
                    for csv_path in csv_files:
                        csv_path.unlink()
                    valid_answer = True
                    print(" | Old files deleted. Proceeding to download...")
                    print("!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!")
                elif user_input == "n":
                    valid_answer = True
                    print(" | Download operation cancelled. Stopping the program.")
                    print("!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!")
                    sys.exit(0)

        # ------- Groups the rows using a dictionary of buffers. -------
        #? defaultdict(): Prevents 'KeyError' exceptions when it is tried to acces or modify
        #   a key that does not yet exist in the dictionary. Instead of raising an error,
        #   it automatically initializes the missing key with a default value.
        buffers = defaultdict(list)

        for row in self.dataset:
            # Handles the string timestamp.
            row_time: Timestamp | str = row["time"]

            # IF the time is stored as a string, converts into a Timestamp.
            if isinstance(row_time, str):
                row_time: Timestamp = pd.to_datetime(arg = row_time)

            key: tuple[int, int] = (row_time.year, row_time.month)
            buffers[key].append(row)

            # Flushes buffer for this year & month, IF it reaches the configured chunk size.
            if len(buffers[key]) >= self.cfg.chunk_size:
                self.__process_and_save_chunk(
                    chunk = buffers[key],
                    year = row_time.year,
                    month = row_time.month
                )
                buffers[key].clear()

        # Flushes any remaining rows left in buffers.
        for key, remaining_chunk in buffers.items():
            if remaining_chunk:
                year, month = key

                self.__process_and_save_chunk(
                    chunk = remaining_chunk,
                    year = year,
                    month = month
                )

    def run_dataloader(self) -> None:
        print("--- Starting with DataLoader... ---")

        # ------- Loads / Downloads the dataset. -------
        if self.cfg.should_download_dataset:
            # Downloads the dataset online from HuggingFace and filters it.
            self._download_online_dataset()
            print("-> Succesfully loaded the dataset from HuggingFaces.")

            self._filter_dataset()
            print("--> Succesfully filtered the dataset with state names.")

            self._export_dataset_as_csv()
            print(f"---> Succesfully exported the dataset into a csv file at location {self.cfg.dataset_csv_files_path}.")
        else:
            # Loads the local dataset.
            self._load_local_dataset()
            
        print("--- Finished with DataLoader. ---")
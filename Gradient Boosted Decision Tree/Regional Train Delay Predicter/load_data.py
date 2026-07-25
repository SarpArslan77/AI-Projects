
#* load_data.py

#TODOs
#   FTH: Fix the Hardcoding.
#   FWT: Fix White Text.

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

# ------- External Libraries. -------
import numpy as np
import pandas as pd
from pandas import Timestamp
from pathlib import Path

#* ======= Config Data Loader Class Initialization =======
@dataclass(frozen = True)
class ConfigDataLoader:
    should_download_dataset: bool
    hugging_face_dataset_path: str
    hugging_face_dataset_data_files: str
    local_dataset_path: str

    selected_german_state: str #TODO Make it so, that it is possible to choose multiple states.
    stations_dir: Path
    stations_json_file_name: str
    date_inervall: tuple[datetime, datetime]

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
            path = self.cfg.hugging_face_dataset_path,
            data_files = self.cfg.hugging_face_dataset_data_files,
            streaming = True
        )

    def _load_local_dataset(self) -> None:
        # ------- Loads the dataset locally from a directory path. -------
        csv_file_dirs_str = str(self.cfg.dataset_csv_files_path / "*.csv")

        self.dataset = load_dataset(
            path = self.cfg.local_dataset_path, #TODO FTH
            data_files = csv_file_dirs_str,
            streaming = True
        )

    def __filter_date(
            self,
            data : dict[str, any],
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
                    date_intervall = self.cfg.date_inervall
                )
        )

        self.dataset = self.dataset["train"] #TODO FTH

    #TODO Re-write this method.
    def _export_dataset_to_csv(self) -> None:
        # ------- Ask to delete old files if the directory exists -------
        dataset_csv_files_path: Path = self.cfg.dataset_csv_files_path

        if dataset_csv_files_path.is_dir():
            # Only prompt if there are actually CSV files to delete
            csv_files = list(dataset_csv_files_path.glob("*.csv"))
            
            if csv_files:
                user_input = input(
                    f"Old CSV files found in '{dataset_csv_files_path}'. "
                    "Do you want to delete them and proceed with the download? (Y/n): "
                ).strip().lower()

                # If user presses Enter, or inputs 'y'/'yes', delete the files
                if user_input in ("", "y", "yes"):
                    for csv_path in csv_files:
                        csv_path.unlink()
                    print("Old files deleted. Proceeding to download...")
                else:
                    print("Operation cancelled. Stopping the program.")
                    sys.exit(0)  # Stops the program cleanly

        # ------- Helper function to process and save a chunk -------
        def _process_and_save_chunk(chunk_data: list[dict], year: int, month: int) -> None:
            if not chunk_data:
                return

            df = pd.DataFrame(chunk_data)

            # Removes unnecessary categorical features.
            df = df.drop(columns=self.cfg.to_be_dropped_categorical_features)

            # Converts timestamp categories into cyclical numerical values.
            for time_col in self.cfg.time_categorical_features:
                dt_series = pd.to_datetime(df[time_col])

                # Mappings of time units to their (series, maximum_period)
                cycles = {
                    "second": (dt_series.dt.second, 60),
                    "minute": (dt_series.dt.minute, 60),
                    "hour":   (dt_series.dt.hour, 24),
                    "day":    (dt_series.dt.day, 31),
                    "month":  (dt_series.dt.month, 12),
                }

                # Dynamically calculate sin and cos for each time unit
                for unit, (series, period) in cycles.items():
                    df[f"{time_col}_{unit}_sin"] = np.round(np.sin(2 * np.pi * series / period), decimals=4)
                    df[f"{time_col}_{unit}_cos"] = np.round(np.cos(2 * np.pi * series / period), decimals=4)

                df[f"{time_col}_year"] = dt_series.dt.year

                # Deletes the old datetime category.
                df = df.drop(columns=time_col)

            # Constructs target file path using parent directory and year_month format.
            csv_file_path = dataset_csv_files_path / f"{year}_{month:02d}.csv"
            
            # Write header only if the file is being created for the very first time.
            write_header = not csv_file_path.exists()

            df.to_csv(
                path_or_buf=csv_file_path,
                mode="a",
                index=False,
                header=write_header
            )

        # ------- Group rows using a dictionary of buffers. -------
        buffers = defaultdict(list)

        for row in self.dataset:
            # Handles string timestamp.
            current_time = row["time"]

            if isinstance(current_time, str):
                current_time = pd.to_datetime(current_time)

            key = (current_time.year, current_time.month)
            buffers[key].append(row)

            # Flushes buffer for this year/month if it reaches the configured chunk size.
            if len(buffers[key]) >= self.cfg.chunk_size:
                _process_and_save_chunk(buffers[key], *key)
                buffers[key].clear()

        # Flushes any remaining rows left in buffers.
        for key, remaining_chunk in buffers.items():
            if remaining_chunk:
                _process_and_save_chunk(remaining_chunk, *key)

    def run_dataloader(self) -> None:
        print("--- Starting with DataLoader... ---")

        # ------- Loads / Downloads the dataset. -------
        if self.cfg.should_download_dataset:
            # Downloads the dataset online from HuggingFace and filters it.
            self._download_online_dataset()
            print("-> Succesfully loaded the dataset from HuggingFaces.")

            self._filter_dataset()
            print("--> Succesfully filtered the dataset with state names.")

            self._export_dataset_to_csv()
            print(f"---> Succesfully exported the dataset into a csv file at location {self.cfg.dataset_csv_files_path}.")
        else:
            # Loads the local dataset.
            self._load_local_dataset()
            
        print("--- Finished with DataLoader. ---")
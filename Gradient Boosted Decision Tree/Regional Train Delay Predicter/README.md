# Regional Train Delay Predictor

A machine learning pipeline to predict train cancellations and delays for German regional passenger trains using Gradient Boosted Decision Trees (CatBoost).

The pipeline uses a two-stage modeling approach:
1. **CatBoost Classifier**: Predicts whether a train connection is canceled or active.
2. **CatBoost Regressor**: For active trains, predicts the change in delay (`delay_delta`) relative to prior station stops, reconstructs total delay, and bins results into operational delay categories.

---

## Features

- **Automated Data Ingestion**: Streams German railway data (`piebro/deutsche-bahn-data` via Hugging Face) or loads existing local monthly CSV files.
- **Geographic & Temporal Filtering**: Filters records by German federal state (e.g., North Rhine-Westphalia) and custom date intervals.
- **Feature Engineering**:
  - Encodes cyclical time components (sin/cos for hour, day, month, etc.).
  - Computes sequential lag features (`prev_delay_in_min`) and delta targets per trip.
- **Smart Parquet Caching**: Automatically saves split datasets to Parquet and skips reprocessing if input files and configurations remain unchanged.
- **Two-Stage CatBoost Architecture**:
  - Handles categorical features natively.
  - Balanced class weighting for cancellation detection.
  - Early stopping on validation metrics.
- **Comprehensive Evaluation & Visualizations**:
  - Bins delay minutes into standard categories (`On Time`, `Minor Delay`, `Moderate Delay`, `Severe Delay`, `Extreme Delay`, and `Canceled`).
  - Plots and saves training/validation loss curves, classification report heatmap, and confusion matrix.
  - Mirrors stdout to terminal logs on disk.

---

## Project Structure

```text
├── main.py                # Main orchestration pipeline (training, inference, evaluation)
├── load_data.py           # Data streaming, filtering by region/date, and monthly chunk exports
├── preprocess_data.py     # Feature engineering, train/val/test splitting, and Parquet caching
├── model.py               # CatBoost Classifier and Regressor wrappers
├── plot_graph.py          # Metric visualization (loss curves, heatmap, confusion matrix)
├── terminal_logger.py     # Custom dual-output stream logger (console + text file)
├── Dataset CSV Files/     # Raw/filtered CSV chunks (generated or user-provided)
├── Other Data Files/      # Lookup files (e.g., all_stations.json, metadata.json)
└── Training Data/         # Output run directories (saved models, plots, logs)

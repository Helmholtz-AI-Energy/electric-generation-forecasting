"""Utility functions module."""

import datetime
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Union
import uuid

import colorlog
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


def set_logger_config(
    level: int = logging.INFO,
    log_file: Union[str, Path] = None,
    log_to_stdout: bool = True,
    colors: bool = True,
) -> None:
    """
    Set up the logger. Should only need to be done once.

    Parameters
    ----------
    level: int, optional
        The logging level. Default is logging.INFO.
    log_file: str | Path, optional
        The file to save the log to.
    log_to_stdout: bool, optional
        True if the log should be printed on stdout. Default is True.
    colors: bool, optional
        Whether to use colored logs (True) or not. Default is True.
    """
    # Get base logger for Propulate.
    base_logger = logging.getLogger("electric-generation-forecasting")
    simple_formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )
    if colors:
        formatter = colorlog.ColoredFormatter(
            fmt="[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s]"
            f"[%(log_color)s%(levelname)s%(reset)s] - %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
        )
        std_handler = logging.StreamHandler(stream=sys.stdout)
        std_handler.setFormatter(formatter)
    else:
        std_handler = logging.StreamHandler(stream=sys.stdout)
        std_handler.setFormatter(simple_formatter)

    if log_to_stdout:
        base_logger.addHandler(std_handler)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(simple_formatter)
        base_logger.addHandler(file_handler)
    base_logger.setLevel(level)


def get_col_diff_intersect(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> Tuple[pd.Index, pd.Index]:
    """
    Return difference and intersection of columns of two dataframes.

    Params
    ------
    df1 : pandas.DataFrame
        The first dataframe.
    df2 : pandas.DataFrame
        The second dataframe.

    Returns
    -------
    pandas.Index
        The difference in columns of the input dataframes.
    pandas.Index
        The intersection of columns of the input dataframes.
    """
    return df1.columns.difference(df2.columns), df1.columns.intersection(df2.columns)


def _correct_time_shift(
    df: pd.DataFrame, usual_length: int = 96
) -> List[List[Union[str, pd.Timestamp]]]:
    """
    Find CET-CEST time shift dates in dataframe index.

    Parameters
    ----------
    df : pandas.DataFrame
        The considered dataframe.
    usual_length : int, optional
        The usual length of one day (96 for 15 min frequency), by default 96.

    Returns
    -------
    List[List[Union[str, pd.Timestamp]]]
        A list of lists with time-shifting dates and their respective lengths.
    """
    # Get unique dates in data index.
    unique_dates = df.index.date.unique().tolist()
    time_shift_dates = []
    for date in unique_dates:
        length = len(df.loc[date])
        if length != usual_length:
            print(f"Time shift at {date}, length is {length}.")
            time_shift_dates.append([str(date), length])
    return time_shift_dates


def plot_actual_predicted_data(actual: pd.DataFrame, predicted: pd.DataFrame) -> None:
    """
    Plot preprocessed load and generation data.

    Parameters
    ----------
    actual : pd.DataFrame
        The actual time series data.
    predicted : pd.DataFrame
        The predicted time series data at the same date time indices.
    """
    fig = make_subplots(
        rows=len(actual.columns),
        cols=1,
        subplot_titles=[
            header.replace("_", " ").capitalize() for header in actual.columns
        ],
    )  # Set up figure with subplots.

    for i, header in enumerate(actual.columns):
        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=actual[header],
                name=header.replace("_", " ").capitalize(),
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=predicted.index,
                y=predicted[header],
                name=header.replace("_", " ").capitalize() + " predicted",
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=actual[header] - predicted[header],
                name=header.replace("_", " ").capitalize() + " residuals",
            ),
            row=i + 1,
            col=1,
        )
        fig.update_yaxes(title_text="Mega Watt", row=i + 1, col=1)
    fig.update_layout(height=10000, width=1200, showlegend=True)
    fig.show()


def calculate_residuals(actual: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate residuals of actual data w.r.t PSLPs.

    Parameters
    ----------
    actual : pd.DataFrame
        The actual time series.
    predicted : pd. DataFrame
        The predicted time series at the same datetime indices.

    Returns
    -------
    pd.DataFrame
        The corresponding residuals.
    """
    return pd.DataFrame(
        {header: actual[header] - predicted[header] for header in actual.columns},
        index=actual.index,
    )


def calculate_errors(
    actual: pd.DataFrame, predicted: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Calculate forecasting errors for ENTSO-E load and generation data.

    Mean absolute error (MAE), mean absolute percentage error (MAPE), and mean squared error (MSE) are considered.

    Parameters
    ----------
    actual : pd.DataFrame
        The actual time series.
    predicted : pd. DataFrame
        The predicted time series at the same datetime indices.

    Returns
    -------
    Dict[str, Dict[str, float]]
        MAE, MAPE, and MSE errors.
    """
    return {
        header: {
            "MAE": mean_absolute_error(actual[header], predicted[header]),
            "MAPE": mean_absolute_percentage_error(actual[header], predicted[header]),
            "MSE": mean_squared_error(actual[header], predicted[header]),
        }
        for header in actual.columns
    }


def construct_output_path(
    output_path: Union[Path, str] = ".",
    output_name: str = "",
    experiment_id: str = None,
    mkdir: bool = True,
) -> Tuple[Path, str]:
    """
    Constructs the path and filename to save results at based on the current time and date.
    Returns an output directory: 'output_path / year / year-month / date / YYYY-mm-dd' or the subdirectory
    'output_path / year / year-month / date / YYYY-mm-dd / experiment_id' if an experiment_id is given and a base_name
    for output files: 'YYYY-mm-dd--HH-MM-SS-<output_name>-<uuid>'.
    All directories on this path are created automatically unless mkdir is set to False.

    Parameters
    ----------
    output_path : Union[pathlib.Path, str]
        The path to the base output directory to create the date-based output directory tree in.
    output_name : str
        Optional label for the csv file, added to the name after the timestamp.
    experiment_id : str
        If this is given, the file is placed in a further subdirectory of that name, i.e.,
        output_path / year / year-month / date / experiment_id / <filename>.csv
        This can be used to group multiple runs of an experiment.
    mkdir : bool
        Whether to create all directories on the output path.

    Returns
    -------
    Tuple[pathlib.Path, str]
        The path to the output directory and the base file name.
    """
    today = datetime.datetime.today()
    path = (
        Path(output_path)
        / str(today.year)
        / f"{today.year}-{today.month}"
        / str(today.date())
    )
    if experiment_id:
        path /= experiment_id
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    base_filename = (
        f"{today.strftime('%Y-%m-%d--%H-%M-%S')}-{output_name}-{str(uuid.uuid4())[:8]}"
    )
    return path, base_filename

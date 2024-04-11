"""Module for ENTSO-E dataset base class."""

import logging
from typing import List, Optional, Tuple

from entsoe import EntsoePandasClient
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import sklearn
import torch

from pslp import calculate_pslps
from utils import set_logger_config

pd.set_option("display.max_rows", 100)
log = logging.getLogger("electric-generation-forecasting")  # Get logger instance.


class EntsoeDataset:
    """
    Fetch and preprocess ENTSO-E load and generation data.

    Attributes
    ----------
    api_key : str
        A valid web token for RESTfulAPI access to the ENTSO-E transparency platform.
    country : str
        The country code.
    df : pandas.DataFrame
        The dataframe holding the considered dataset.
    downsample : bool
        Whether to downsample the data to 1h resolution (True) or not (False).
    end_date : str
        The end date.
    start_date : str
        The start date
    time_zone : str
        The time zone.

    Methods
    -------
    fetch_data()
        Fetch load and generation per production type from ENTSO-E transparency platform for requested time interval.
    plot_data()
        Plot preprocessed load and generation data.
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        api_key: str,
        country_code: Optional[str] = "10Y1001A1001A83F",
        time_zone: Optional[str] = "Europe/Berlin",
        downsample: Optional[bool] = False,
        drop_consumption: Optional[bool] = True,
    ) -> None:
        """
        Initialize a basic ENTSO-E load and generation per production type dataset.

        Parameters
        ----------
        start_date : str
            The overall start date as "YYYYMMDD".
        end_date : str
            The overall end date as "YYYYMMDD".
        api_key : str
            The web token for RESTful API access to ENTSO-E transparency platform.
        country_code : str, optional
            The country code. Default is Germany.
        time zone : str, optional
            The time zone. Default is "Europe/Berlin"
        downsample : bool, optional
            True if data should be down-sampled to 1h resolution. Default it True.
        drop_consumption : bool, optional
            True if columns containing actual consumption should be dropped.
            Default is True.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        self.country = country_code
        self.time_zone = time_zone
        self.downsample = downsample
        self.df = self.fetch_data(drop_consumption=drop_consumption)

    def _load_data(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Load actual load and actual aggregated generation per production type for requested time interval.

        Parameters
        ----------
        start_date : str
            The start date as "yyyymmdd".
        end_date : str
            The end date as "yyyymmdd".

        Returns
        -------
        pandas.DataFrame
            Dataframe with time points as indices and load + generation per type as columns.
        """
        # Initialize client and settings.
        client = EntsoePandasClient(api_key=self.api_key)
        start = pd.Timestamp(start_date, tz=self.time_zone)
        end = pd.Timestamp(end_date, tz=self.time_zone)
        # Query data and save to dataframe.
        df_load = client.query_load(self.country, start=start, end=end)
        log.info(f"Actual load has shape {df_load.shape}.")
        df_gen = client.query_generation(
            self.country, start=start, end=end, psr_type=None
        )
        df_gen.columns = [" ".join(a) for a in df_gen.columns.to_flat_index()]
        log.info(f"Actual generation per production type has shape {df_gen.shape}.")
        df_final = pd.concat(
            [df_load, df_gen], axis=1
        )  # Concatenate dataframes in columns dimension.
        log.info(f"Concatenated data frame has shape {df_final.shape}.")

        return df_final.astype("float")

    def fetch_data(
        self,
        drop_consumption: Optional[bool] = True,
    ):
        """
        Fetch load and generation per production type from ENTSO-E transparency platform for requested time interval.

        Parameters
        ----------
        drop_consumption : bool, optional
            True if columns containing actual consumption should be dropped.
            Default is True.
        """
        # Define mapping for generation technologies from ENTSO-E to the classification used in this study.
        # Unnewehr et al. (2022), Open-data based carbon emission intensity signals for electricity generation in
        # European countries – top down vs. bottom up approach, Cleaner Energy Systems, Volume 3, 2022,
        # doi: 10.1016/j.cles.2022.100018
        generation_type_mapping = {
            "actual_load": ["Actual Load"],
            "hard_coal": ["Fossil Hard coal Actual Aggregated"],
            "lignite": ["Fossil Brown coal/Lignite Actual Aggregated"],
            "gas": ["Fossil Gas Actual Aggregated"],
            "other_fossil": [
                "Fossil Coal-derived gas Actual Aggregated",
                "Fossil Oil Actual Aggregated",
                "Other Actual Aggregated",
            ],
            "nuclear": ["Nuclear Actual Aggregated"],
            "biomass": ["Biomass Actual Aggregated"],
            "waste": ["Waste Actual Aggregated"],
            "other_renewable": [
                "Geothermal Actual Aggregated",
                "Other renewable Actual Aggregated",
            ],
            "hydro": [
                "Hydro Pumped Storage Actual Aggregated",
                "Hydro Run-of-river and poundage Actual Aggregated",
                "Hydro Water Reservoir Actual Aggregated",
            ],
            "solar": [
                "Solar Actual Aggregated",
            ],
            "wind_onshore": ["Wind Onshore Actual Aggregated"],
            "wind_offshore": ["Wind Offshore Actual Aggregated"],
        }
        try:
            log.info(f"Loading data from {self.start_date} to {self.end_date}")
            df = self._load_data(start_date=self.start_date, end_date=self.end_date)
        except Exception as e:
            log.info("FAILED.", e)

        log.info(
            "Setting N/A's to 0 for 'Nuclear' due to shutdown of nuclear power plants in GER..."
        )

        df["Nuclear Actual Aggregated"] = df["Nuclear Actual Aggregated"].fillna(0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(tz="UTC+01:00")
        if drop_consumption:  # Drop columns containing actual consumption.
            log.info("Dropping columns containing consumption...")
            df.drop(list(df.filter(regex="Consumption")), axis=1, inplace=True)
        else:
            pass
            # TODO: Possibly implement difference between columns production - consumption if available.

        # TODO: How to handle NaNs? Interpolate? Use PSLP? Set to 0?
        df.interpolate(method="time", axis=0, inplace=True)
        # Time-based interpolation is specifically designed for time series data. It fills missing values using
        # linear interpolation based on time, where the time difference between consecutive data points is used
        # to compute intermediate values.

        # Apply generation type mapping (Unnewehr et al. 2022).
        log.debug(f"Original categories in the data are: {df.columns}")
        for joint_category, old_categories in generation_type_mapping.items():
            existing_columns = [col for col in old_categories if col in df.columns]
            log.debug(f"Existing columns are {existing_columns}.")
            if existing_columns:
                # Sum up existing columns and drop them
                df[joint_category] = df[existing_columns].sum(axis=1, skipna=False)
                df.drop(columns=existing_columns, inplace=True)

        if self.downsample:
            log.info("Downsample to 1h resolution...")
            df = df.resample("1h").mean()

        log.info(f"Returning final data frame of shape {df.shape}...")
        return df

    def plot_data(self) -> None:
        """Plot preprocessed load and generation data."""
        fig = make_subplots(
            rows=self.df.columns.size,
            cols=1,
            subplot_titles=[
                header.replace("_", " ").capitalize() for header in self.df.columns
            ],
        )

        for i, header in enumerate(self.df.columns):
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df[header],
                    name=header.replace("_", " ").capitalize(),
                ),
                row=i + 1,
                col=1,
            )
        fig.update_layout(height=10000, width=1200)
        fig.show()


class ScaledEntsoeDataset(EntsoeDataset):
    """
    Fetch, preprocess, and scale ENTSO-E load and generation data.

    Attributes
    ----------
    scaled_df : pandas.DataFrame
        The scaled dataframe.
    scaler : sklearn.preprocessing.MinMaxScaler
        The scaler used.

    Methods
    -------
    scale_data()
        Scale input data.
    unscale_data()
        Un-scale input data.

    Notes
    -----
    This class inherits attributes and methods from `EntsoeDataset`.

    See Also
    --------
    :class:`EntsoeDataset`
        The parent class from which this class inherits.
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        api_key: str,
        country_code: Optional[str] = "10Y1001A1001A83F",
        time_zone: Optional[str] = "Europe/Berlin",
        downsample: Optional[bool] = False,
        drop_consumption: Optional[bool] = True,
    ) -> None:
        """
        Initialize a scaled ENTSO-E load and generation per production type dataset.

        Parameters
        ----------
        start_date : str
            The overall start date as "YYYYMMDD".
        end_date : str
            The overall end date as "YYYYMMDD".
        api_key : str
            The web token for RESTful API access to ENTSO-E transparency platform.
        country_code : str, optional
            The country code. Default is Germany.
        time zone : str, optional
            The time zone. Default is "Europe/Berlin"
        downsample : bool, optional
            True if data should be down-sampled to 1h resolution. Default it True.
        drop_consumption : bool, optional
            True if columns containing actual consumption should be dropped.
            Default is True.
        """
        super().__init__(
            start_date,
            end_date,
            api_key,
            country_code,
            time_zone,
            downsample,
            drop_consumption,
        )
        self.scaled_df, self.scaler = self.scale_data(self.df)

    @staticmethod
    def scale_data(
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, sklearn.preprocessing.MinMaxScaler]:
        """
        Scale input data.

        Returns
        -------
        pandas.DataFrame
            The scaled data.
        sklearn.preprocessing.MinMaxScaler
            The scaler.
        """
        log.info("Scale features globally.")

        # Copy original data frame for storing scaled data.
        scaled_df = df.copy()

        # Fit MinMaxScaler to the entire dataset
        scaler = sklearn.preprocessing.MinMaxScaler().fit(
            df.to_numpy().astype(float).reshape(-1, 1)
        )

        # Scale features.
        for column in scaled_df.columns:
            scaled_df[column] = scaler.transform(
                df[column].to_numpy(dtype=float).reshape(-1, 1)
            )
        return scaled_df, scaler

    def plot_data(self, use_scaled: Optional[bool] = True) -> None:
        """
        Plot preprocessed load and generation data.

        Parameters
        ----------
        use_scaled : bool, optional
            Whether to plot scaled data (True) or raw data (False).
            Default is True.
        """
        df = self.scaled_df if use_scaled else self.df

        fig = make_subplots(
            rows=df.columns.size,
            cols=1,
            subplot_titles=[
                header.replace("_", " ").capitalize() for header in df.columns
            ],
        )

        for i, header in enumerate(df.columns):
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df[header], name=header.replace("_", " ").capitalize()
                ),
                row=i + 1,
                col=1,
            )
        fig.update_layout(height=10000, width=1200)
        fig.show()

    @staticmethod
    def unscale_data(
        df: pd.DataFrame,
        scaler: sklearn.preprocessing.MinMaxScaler,
    ) -> pd.DataFrame:
        """
        Un-scale input data.

        Parameters
        ----------
        df : pandas.DataFrame
            The data to un-scale.
        scaler : sklearn.preprocessing.MinMaxScaler
            The scaler to use.

        Returns
        -------
        pandas.DataFrame
            Un-scaled data.
        """
        log.info("Un-scale features globally.")
        unscaled_df = df.copy()  # Copy original data frame for storing scaled data.
        for column in df.columns:
            unscaled_df[column] = scaler.inverse_transform(
                df[column].to_numpy(dtype=float).reshape(-1, 1)
            )
        return unscaled_df


class ResidualEntsoeDataset(ScaledEntsoeDataset):
    """
    Construct residual dataset from (scaled) ENTSO-E load and generation data w.r.t. PSLPs.

    Attributes
    ----------
    residual_df : pandas.DataFrame
        The residual dataset.
    residual_scaled_df : pandas.DataFrame
        The residual scaled dataset.

    Notes
    -----
    This class inherits attributes and methods from `ScaledEntsoeDataset`.

    See Also
    --------
    :class:`ScaledEntsoeDataset`
        The parent class from which this class inherits.
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        api_key: str,
        country_code: Optional[str] = "10Y1001A1001A83F",
        time_zone: Optional[str] = "Europe/Berlin",
        downsample: Optional[bool] = False,
        drop_consumption: Optional[bool] = True,
        lookback: Optional[int] = 3,
    ) -> None:
        """
        Initialize a scaled ENTSO-E load and generation per production type dataset.

        Parameters
        ----------
        start_date : str
            The overall start date as "YYYYMMDD".
        end_date : str
            The overall end date as "YYYYMMDD".
        api_key : str
            The web token for RESTful API access to ENTSO-E transparency platform.
        country_code : str, optional
            The country code. Default is Germany.
        time zone : str, optional
            The time zone. Default is "Europe/Berlin"
        downsample : bool, optional
            True if data should be down-sampled to 1h resolution. Default it True.
        drop_consumption : bool, optional
            True if columns containing actual consumption should be dropped.
            Default is True.
        lookback : int, optional
            The number of days to consider in each category for calculating PSLP.
            Default is 3.
        """
        super().__init__(
            start_date,
            end_date,
            api_key,
            country_code,
            time_zone,
            downsample,
            drop_consumption,
        )
        self.pslp_df = calculate_pslps(self.df, lookback=lookback, country_code="DE")
        self.pslp_scaled_df = calculate_pslps(
            self.scaled_df, lookback=lookback, country_code="DE"
        )
        self.residual_df = self.df - self.pslp_df
        self.residual_scaled_df = self.scaled_df - self.pslp_scaled_df

    def plot_data(self, use_scaled: Optional[bool] = True) -> None:
        """
        Plot preprocessed load and generation data.

        Parameters
        ----------
        use_scaled : bool, optional
            Whether to plot scaled data (True) or raw data (False).
            Default is True.
        """
        if use_scaled:
            df = self.scaled_df
            pslps = self.pslp_scaled_df
            residuals = self.residual_scaled_df
        else:
            df = self.df
            pslps = self.pslp_df
            residuals = self.residual_df
        fig = make_subplots(
            rows=df.columns.size,
            cols=1,
            subplot_titles=[
                header.replace("_", " ").capitalize() for header in df.columns
            ],
        )

        for i, header in enumerate(df.columns):
            fig.add_trace(
                go.Scatter(x=df.index, y=df[header], name=header.replace("_", " ")),
                row=i + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=pslps.index,
                    y=pslps[header],
                    name=f"{header.replace('_',' ').capitalize()} PSLPs ",
                ),
                row=i + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=residuals.index,
                    y=residuals[header],
                    name=f"{header.replace('_',' ').capitalize()} residuals",
                ),
                row=i + 1,
                col=1,
            )
        fig.update_layout(height=10000, width=1200)
        fig.show()


class SequenceDataset:
    """
    PyTorch dataset for sequenced time series data.

    Attributes
    ----------
    df : pandas.DataFrame
        Continuous time series to generate sequences from.
    samples : torch.Tensor
        The sorted samples.
    targets : torch.Tensor
        The sorted targets.

    Methods
    -------
    __getitem__()
        Get one item (sample-target pair) at specified index position in dataset.
    __len__()
        Get number of samples in the dataset.
    _generate_sequences()
        Generate sequences from continuous time series dataframe.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        training_window: Optional[int] = 168,
        prediction_window: Optional[int] = 1,
        stride: Optional[int] = 1,
    ) -> None:
        """
        Construct a sequenced dataset from a time series dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            The continuous time series data.
        training_window : int
            The training window, i.e., the number of previous time steps used as input features.
            Default is 168, i.e., one week for hourly resolved data.
        prediction_window : int
            The prediction window, i.e., the number of future time steps to predict.
            Default is 1, i.e., predict the next hour for hourly resolved data.
        stride : int
            The step size between consecutive training sequences. Default is 1.
        """
        self.df = df
        self.training_window = training_window
        self.prediction_window = prediction_window
        self.stride = stride
        self.samples, self.targets, self.target_datetime_indices = (
            self.generate_sequences()
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get one item (sample-target pair) at specified index position in dataset.

        Parameters
        ----------
        idx : int
            The index position

        Returns
        -------
        torch.Tensor
            The sample at the given index position.
        torch.Tensor
            The corresponding target (at the same index position).
        """
        return self.samples[idx], self.targets[idx]

    def __len__(self):
        """
        Get length of dataset, i.e., number of labeled sample-target pairs.

        Returns
        -------
        int
            The number of labeled samples in the dataset.
        """
        assert len(self.samples) == len(self.targets)
        return len(self.samples)

    def generate_sequences(self) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """
        Create sequences from univariate time series.

        Returns
        -------
        torch.Tensor
            The samples
        torch.Tensor
            The targets.
        torch.Tensor
            The datetime index time points for each sample.
        """
        samples, targets, targets_datetime_indices = [], [], []
        log.debug(f"Generate sequences from data at {self.df.columns}.")
        # Loop through continuous time series data to cut out sequences as labeled samples.
        for i in range(
            0,
            len(self.df) - self.training_window - self.prediction_window + 1,
            self.stride,
        ):
            samples.append(
                self.df[i : i + self.training_window].to_numpy(dtype=float)
            )  # Slice all columns for input sequence.
            targets.append(
                self.df[
                    i
                    + self.training_window : i
                    + self.training_window
                    + self.prediction_window
                ].to_numpy(dtype=float)
            )  # Slice all columns for target sequence
            targets_datetime_indices.append(
                self.df.index[
                    i
                    + self.training_window : i
                    + self.training_window
                    + self.prediction_window
                ]  # Index of the last time point in the training window
            )
        samples, targets = np.array(samples), np.array(targets)
        # Return tensors of samples, targets, and datetime index time points.
        return (
            torch.tensor(samples, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
            targets_datetime_indices,
        )


if __name__ == "__main__":
    set_logger_config(
        level=logging.INFO,  # logging level
        log_file="./entsoe_dataset.log",  # logging path
        log_to_stdout=True,  # Print log on stdout.
        colors=True,  # Use colors.
    )  # Set up logger.

    api_key = ...  # Insert your web token for RESTful API here!
    country_code = "10Y1001A1001A83F"  # Germany
    time_zone = "Europe/Berlin"  # Time zone for Germany
    start_date = "20180101"
    end_date = "20180601"
    downsample = False
    drop_consumption = True
    lookback = 3

    entsoe_dataset = EntsoeDataset(
        start_date=start_date,
        end_date=end_date,
        api_key=api_key,
        country_code=country_code,
        time_zone=time_zone,
        downsample=downsample,
        drop_consumption=drop_consumption,
    )
    print("Actual Load is:", entsoe_dataset.df["actual_load"])
    entsoe_dataset.plot_data()

    scaled_dataset = ScaledEntsoeDataset(
        start_date=start_date,
        end_date=end_date,
        api_key=api_key,
        country_code=country_code,
        time_zone=time_zone,
        downsample=downsample,
        drop_consumption=drop_consumption,
    )

    print("Actual Load is:", scaled_dataset.scaled_df["actual_load"])
    unscaled_df = ScaledEntsoeDataset.unscale_data(
        scaled_dataset.scaled_df, scaled_dataset.scaler
    )
    print(
        "Values still close after scaling and unscaling back and forth?",
        torch.allclose(
            torch.tensor(scaled_dataset.df.values, dtype=torch.float64),
            torch.tensor(
                unscaled_df.values,
                dtype=torch.float64,
            ),
        ),
    )
    scaled_dataset.plot_data(use_scaled=True)

    residual_dataset = ResidualEntsoeDataset(
        start_date=start_date,
        end_date=end_date,
        api_key=api_key,
        country_code=country_code,
        time_zone=time_zone,
        downsample=downsample,
        drop_consumption=drop_consumption,
        lookback=lookback,
    )

    residual_dataset.plot_data(use_scaled=True)
    residual_dataset.plot_data(use_scaled=False)

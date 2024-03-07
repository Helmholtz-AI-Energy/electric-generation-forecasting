"""SARIMA module."""

import datetime
import logging
import pathlib
import pickle
import time
from typing import Dict, Union
import uuid

import pandas as pd
from pmdarima.arima.utils import ndiffs, nsdiffs
from pmdarima.arima import auto_arima, ARIMA

from entsoe_dataset import EntsoeDataset
from utils import set_logger_config, plot_actual_predicted_data

pd.set_option("display.max_rows", 100)
log = logging.getLogger("electric-generation-forecasting")  # Get logger instance.


# TODO: Add option for in-sample prediction with `model.predict_in_sample()`. Predicts the original training (in-sample)
#  time series values. This can be useful when wanting to visualize the fit, and qualitatively inspect the efficacy of
#  the model, or when wanting to compute the residuals of the model.
def fit_sarima(
    dataset: EntsoeDataset,
    save_dir: Union[pathlib.Path, str] = "./",
    update_level=2,
    model_ckpt: Union[pathlib.Path, str, None] = None,
    min_p: int = 1,
    min_q: int = 1,
    max_p: int = 3,
    max_q: int = 3,
    min_ps: int = 1,
    min_qs: int = 1,
    max_ps: int = 2,
    max_qs: int = 2,
) -> Dict[str, ARIMA]:
    """
    Fit SARIMA model to data.

    Parameters
    ----------
    dataset : EntsoeDataset
        The ENTSO-E dataset to fit the SARIMA model to.
    save_dir : Union[pathlib.Path, str], optional
        The path to the directory where to save the models to. Default is current working directory.
    update_level : int, optional
        The update level.
        0 - Update current model with new observations.
        1 - Refit model of known (seasonal) order.
        2 - Find completely new model with `auto_arima` (default).
    model_ckpt : str, optional
        The path to the SARIMA model checkpoint dictionary.
    min_p : int, optional
        The starting value of p, the order (or number of time lags) of the autoregressive (“AR”) model.
        Must be a positive integer. Default is 1.
    min_q : int, optional
        The starting value of q, the order of the moving-average (“MA”) model. Must be a positive integer.
        Default is 1.
    max_p : int, optional
        The maximum value of p, inclusive. Must be a positive integer greater than or equal to `start_p`.
        Default is 3.
    max_q : int, optional
        The maximum value of q, inclusive. Must be a positive integer greater than `start_q`. Default is 3.
    min_ps : int, optional
        The starting value of P, the order of the autoregressive portion of the seasonal model. Default is 1.
    min_qs : int, optional
        The starting value of Q, the order of the moving-average portion of the seasonal model. Default is 1.
    max_ps : int, optional
        The maximum value of P, inclusive. Must be a positive integer greater than `start_ps`. Default is 2.
    max_qs : int, optional
        The maximum value of Q, inclusive. Must be a positive integer greater than `start_qs`. Default is 2.

    Returns
    -------
    Dict[str, ARIMA]
        The fitted SARIMA model (value) for each generation type (key).
    """
    # Determine seasonal cycle for SARIMA model. Each day is a season, i.e., 24 * 1h or 96 * 15min.
    seasonal_cycle = 24 if dataset.downsample else 96

    models = {}  # Initialize dict to store trained SARIMA models.

    if model_ckpt is not None:  # Check if specified load dir exists.
        if pathlib.Path.exists(pathlib.Path(model_ckpt)) is False:
            log.warning(
                "WARNING: Specified loading directory does not exist. Fitting models from scratch."
            )
            update_level = 2  # Set update level to 2, i.e., find completely new model with `auto_arima`.

    pathlib.Path(save_dir).mkdir(
        parents=True, exist_ok=True
    )  # Create directory for saving checkpoints to.

    if update_level == 0:  # Update current model with new observations.
        with open(model_ckpt, "rb") as f:
            sarima_models = pickle.load(f)
        log.info("Update existing models with new observations...")
        # Loop over categories in data, i.e., load and generation types.
        for category, model in sarima_models.items():
            log.info(f"Update model for {category}.")
            start_time = time.perf_counter()
            model.update(dataset.df[category])
            duration = time.perf_counter() - start_time
            log.info(f"DONE: Updating model for {category} took {duration} s.")
            models[category] = model
            log.info(model.summary())

    elif update_level == 1:
        log.info("Refit models of known (seasonal) order on given data...")
        with open(model_ckpt, "rb") as f:
            sarima_models = pickle.load(f)
        # Loop over categories in data, i.e., load and generation types.
        for category, old_model in sarima_models.items():
            log.info(f"Refit model for {category}.")
            start_time = time.perf_counter()
            model = ARIMA(
                order=old_model.order, seasonal_order=old_model.seasonal_order
            )
            model.fit(dataset.df[category])
            duration = time.perf_counter() - start_time
            log.info(f"DONE: Refitting model for {category} took {duration} s.")
            models[category] = model
            print(model.summary())

    elif update_level == 2:
        log.info("Fit new models from scratch with `auto_arima`...")
        # Loop over categories in data, i.e., load and generation types.
        for category in dataset.df.columns:
            log.info(
                f"Consider {category}.\nPre-compute (seasonal) differencing order to accelerate auto-ARIMA..."
            )
            # Estimate order of differencing d by performing a stationarity test for different d's.
            # Selects max. value d for which time series is judged stationary by statistical test.
            # Default unit root test of stationarity: Kwiatkowski–Phillips–Schmidt–Shin (KPSS)
            start_time = time.perf_counter()
            d = ndiffs(dataset.df[category], test="kpss")
            # Estimate order of seasonal differencing D by performing stationarity test of seasonality for
            # different D's. Selects max. value D for which time series is judged seasonally stationary by
            # statistical test. Default unit root test of stationarity: Osborn-Chui-Smith-Birchenhall (OCSB)
            ds = nsdiffs(dataset.df[category], m=seasonal_cycle, test="ocsb")
            log.info(
                f"Differencing order is {d}. Seasonal differencing order is {ds}.\n"
                f"Automatically discover optimal order for SARIMAX model for {category}..."
            )
            model = auto_arima(
                dataset.df[category],
                start_p=min_p,
                d=d,
                start_q=min_q,
                max_p=max_p,
                max_q=max_q,
                start_P=min_ps,
                start_Q=min_qs,
                max_P=max_ps,
                max_Q=max_qs,
                D=ds,
                m=seasonal_cycle,
                trace=True,
            )
            duration = time.perf_counter() - start_time
            log.info(
                f"DONE: Fitting model for {category} from scratch took {duration} s."
            )

            models[category] = model
            log.info(model.summary())

    log.info("DONE.")
    today = datetime.datetime.today()
    base_filename = f"{today.strftime('%Y-%m-%d--%H-%M-%S')}-{str(uuid.uuid4())[:8]}"
    with open(
        pathlib.Path(save_dir) / f"arima_models_dict-{base_filename}.pkl", "wb"
    ) as pkl:
        pickle.dump(models, pkl)
    return models


def predict_sarima(
    models: Dict[str, ARIMA],
    start_date: str,
    end_date: str,
    frequency="h",
    time_zone="Europe/Berlin",
) -> pd.DataFrame:
    """
    Generate forecasts using fitted SARIMA models.

    Parameters
    ----------
    models : Dict[str, ARIMA]
        Fitted SARIMA models (value) for each generation type (key).
    start_date : str
        Start date for the forecast.
    end_date : str
        End date for the forecast.
    frequency : str
        Frequency of the forecast (default='H' for hourly).
    time_zone : str
        The time zone.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the forecast values for each model.
    """
    forecast_results = {}  # Dict to store forecasted values for each model.
    forecast_index = pd.date_range(
        start=start_date, end=end_date, freq=frequency, tz=time_zone
    )
    # Generate forecast for each model, i.e., each generation type.
    for generation_type, model in models.items():
        forecast_results[generation_type] = model.predict(forecast_index.size)

    # Create dataframe for the forecasted values.
    forecast_df = pd.DataFrame.from_dict(
        forecast_results,
    )
    forecast_df.index = forecast_index
    return forecast_df


if __name__ == "__main__":
    set_logger_config(
        level=logging.INFO,  # logging level
        log_file="./sarima.log",  # logging path
        log_to_stdout=True,  # Print log on stdout.
        colors=True,  # Use colors.
    )  # Set up logger.

    api_key = "6e68642c-8403-4caa-af31-bda40b8c67f6"  # Web token for RESTful API
    country_code = "10Y1001A1001A83F"  # Germany
    time_zone = "Europe/Berlin"  # Time zone for Germany
    start_date = "20180601"
    end_date = "20180701"
    downsample = True
    drop_consumption = True
    save_dir = "./"
    update_level = 2
    model_ckpt = "arima_models_dict.pkl"

    frequency = "1h" if downsample else "15min"

    train_dataset = EntsoeDataset(
        start_date=start_date,
        end_date=end_date,
        api_key=api_key,
        country_code=country_code,
        time_zone=time_zone,
        downsample=downsample,
        drop_consumption=drop_consumption,
    )

    sarima_models = fit_sarima(train_dataset, save_dir, update_level, model_ckpt)

    forecast_start_date = "20180701"
    forecast_end_date = "20180708"

    test_dataset = EntsoeDataset(
        start_date=forecast_start_date,
        end_date=forecast_end_date,
        api_key=api_key,
        country_code=country_code,
        time_zone=time_zone,
        downsample=downsample,
        drop_consumption=drop_consumption,
    )

    forecast_result = predict_sarima(
        sarima_models, forecast_start_date, forecast_end_date, frequency, time_zone
    )
    log.info("Forecast values:", forecast_result)
    plot_actual_predicted_data(test_dataset.df, forecast_result)

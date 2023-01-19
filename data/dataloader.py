import torch
import pandas as pd
import entsoe
import holidays

from torch.utils.data import Dataset
from pandas import DataFrame

class TimeSeriesDataset(Dataset):
    def __init__(
            self,
            data: DataFrame,
            historic_window: int,
            forecast_window: int,
        )

    # Input data shape
    self.historic_window = historic_window
    self.forecast_window = forecast_window
    self.total_window = self.historic_window + self.forecast_window
    
    self.dataframe = data

    # self.unnormalized_data = torch.tensor(data)
    # Normalize data ! (Between 0, 1 seems to work best according to Arvid)
    # Reshape data into cycles (days).

    @classmethod
    def from_csv(
            cls, 
            file,
            historic_window: int,
            forecast_window: int,
            pslp_cycles: int = 3
            country: Optional[str] = None,
            downsample_rate: Optional[int] = None,
        )

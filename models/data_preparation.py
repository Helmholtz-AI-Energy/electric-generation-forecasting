import pandas as pd

def prepare_data(sector, start, end):

    if sector == "load":

        data = pd.DataFrame()
        data = pd.read_csv("./data/load_data.csv",
                           header=[0],
                           index_col=0,
                           parse_dates=True
                           )

        data.index = pd.to_datetime(data.index,
                                    utc=True
                                    )
        print(data.head())
        data = data.resample('15Min').last()
        # create empty DataFrames

    scores = pd.DataFrame()

    return data, scores


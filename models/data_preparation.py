import numpy as np

import pandas as pd
import pytz
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
                                    ).tz_convert(tz="Europe/Berlin")
        print(data.head())
        data = data.resample('15Min').last()
        # create empty DataFrames

    elif sector == "generation":
        data = pd.DataFrame()
        data = pd.read_csv("./data/entsoe_gen.csv",
                           header=[0],
                           index_col=0,
                           parse_dates=True,
                           low_memory=False,
                           ).drop(['Fossil Gas.1',
                                   'Fossil Oil.1',
                                   'Hydro Water Reservoir.1',
                                   'Nuclear.1',
                                   'Other renewable.1',
                                   'Solar.1',
                                   'Wind Onshore.1'],
                                  axis=1,
                                  )

        data.drop(index=data.index[0], axis=0, inplace=True)

        data.index = pd.to_datetime(data.index,
                                    utc=True
                                    ).tz_convert(tz="Europe/Berlin")

        #germ = pytz.timezone('Europe/Berlin')
        #data.index = data.index.tz_localize(pytz.utc).tz_convert(germ)

        print(data.head())
        data = data.resample('15Min').last().astype(float)



    scores = pd.DataFrame()

    return data[start:end], scores


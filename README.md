# Electric Generation Forecasting (EGF)
## AI-based Prediction of the German Electricity Mix

This repository deals with AI-based prediction of the German electricity mix.

### Problem
We want to perform time series prediction of the ENTSO-E actual generation per production type and actual load data, 
i.e., predict the values of these quantities as a "vector" for the next time step(s) from historic data. This basically 
is multi output regression. We are interested in the short-term load forecasting regime (from 24 h to one week). STLF is 
essential for controlling and scheduling of the power system in making everyday power system operation, interchange 
evaluation, security assessment, reliability analysis, and spot price calculation, which leads to the higher accuracy 
requirement compared to long-term prediction.

### Data
We consider the actual generation per production type and the actual load in Germany from the [ENTSO-E 
transparency platform](https://transparency.entsoe.eu/). The data is published daily and has a 15-min frequency. 
**After applying for your personal security token**, you can download the data via the RESTful API from the ENTSO-E 
transparency platform. We use the [entsoe-py Python package](https://github.com/EnergieID/entsoe-py) as a Python client for this.

From the [official documentation](https://eur-lex.europa.eu/LexUriServ/LexUriServ.do?uri=OJ:L:2013:163:0001:0012:EN:PDF):

>#### Actual Generation per Production Type
>Actual aggregated net generation output (MW) per market time unit and per production type.
The information shall be published no later than one hour after the operational period.

>**Specification of calculation** Average of all available instantaneous net generation output values on each market time unit. If a net generation output is not known, it shall be estimated. The actual generation of small-scale units might be estimated if no real-time measurement devices exist.

>**Primary owner of the data** Owners of generation units or transmission system operators (TSOs)  
>**Data provider** TSOs or other Data Provider of information depending on local organisation.

#### Manual inspection of exemplary data for GER downloaded as CSV file 

In an exemplary data snippet downloaded for Germany, there are 23 columns, i.e., raw features:
>- A. Area: All the same because we are only interested in Germany (DE).
>- B. MTU: Considered time window (start + end with delta = 15 min)
>- C. Biomass - Actual Aggregated [MW]
>- D. Fossil Brown coal/Lignite - Actual Aggregated [MW]
>- E. Fossil Coal-derived gas - Actual Aggregated [MW]
>- F. Fossil Gas - Actual Aggregated [MW]
>- G. Fossil Hard coal - Actual Aggregated [MW]
>- H. Fossil Oil - Actual Aggregated [MW]
>- I. Fossil Oil shale - Actual Aggregated [MW]
>- J. Fossil Peat - Actual Aggregated [MW]
>- K. Geothermal - Actual Aggregated [MW]
>- L. Hydro Pumped Storage - Actual Aggregated [MW]
>- *M. Hydro Pumped Storage  - Actual Consumption [MW]* WHAT IS THIS?
>- N. Hydro Run-of-river and poundage - Actual Aggregated [MW]
>- O. Hydro Water Reservoir - Actual Aggregated [MW]
>- P. Marine - Actual Aggregated [MW]
>- Q. Nuclear - Actual Aggregated [MW]
>- R. Other - Actual Aggregated [MW]
>- S. Other renewable - Actual Aggregated [MW]
>- T. Solar - Actual Aggregated [MW]
>- U. Waste - Actual Aggregated [MW]
>- V. Wind Offshore - Actual Aggregated [MW]
>- W. Wind Onshore - Actual Aggregated [MW]

To reduce the problem complexity, we apply the category mapping from Unnewehr et al. (2022), 
*Open-data based carbon emission intensity signals for electricity generation in European countries – 
top down vs. bottom up approach*, Cleaner Energy Systems, Volume 3, 2022, doi: [10.1016/j.cles.2022.100018](https://doi.org/10.1016/j.cles.2022.100018):
>- Actual Load: Actual Load
>- Hard Coal: Fossil Hard coal Actual Aggregated
>- Lignite: Fossil Brown coal/Lignite Actual Aggregated
>- Gas: Fossil Gas Actual Aggregated
>- Other Fossil: Fossil Coal-derived gas Actual Aggregated, Fossil Oil Actual Aggregated, Other Actual Aggregated
>- Nuclear: Nuclear Actual Aggregated
>- Biomass: Biomass Actual Aggregated
>- Waste: Waste Actual Aggregated
>- Other Renewable: Geothermal Actual Aggregated, Other renewable Actual Aggregated
>- Hydro: Hydro Pumped Storage Actual Aggregated, Hydro Run-of-river and poundage Actual Aggregated, Hydro Water Reservoir Actual Aggregated
>- Solar: Solar Actual Aggregated
>- Wind Onshore: Wind Onshore Actual Aggregated
>- Wind Offshore: Wind Offshore Actual Aggregated

### Models

[https://www.entsoe.eu/Technopedia/techsheets/enhanced-load-forecasting](https://www.entsoe.eu/Technopedia/techsheets/enhanced-load-forecasting)
#### Statistical methods

* **Persistence**
    * Used as a baseline model / reference for evaluating a model's predictive power / performance. 
    * Useful to know if a forecast model provides better results than a trivial reference model, here the persistence model.
    * Persistence represents probably the simplest way of producing a forecast. A persistence model assumes that the future 
      value of a time series is calculated under the assumption that nothing changes between the current time and the forecast time. 
      In terms of the electricity mix, the persistence model estimates that the mix at time t+1 equals the mix at time t.


* **Standard load profiles (SLPs)**
    * https://ieeexplore.ieee.org/document/9221967
    * 3 different categories: weekday, Saturday, Sun-/Holiday
    * Use category-based average of last n profiles as forecast: Last(n) category-forecast
    * Usually: n = 3 (can also capture seasonal + weather related changes)
    * Last(1) = persistence (tomorrow will be the same as today), 
      simple + efficient, but struggles with weekends + holidays.
    * Useful Python package: `holidays`


* **ARIMA (Auto-Regressive Integrated Moving Average) model**
    * (Partial) Autocorrelation Function (P)ACF
      * https://www.youtube.com/watch?v=DeORzP0go5I
      * https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/
    * [Stationarity](https://www.youtube.com/watch?v=oY-j2Wof51c&list=PLvcbYUQ5t0UHOLnBzl46_Q6QKtFgfMGc3&index=4)
    * Autoregressive Model AR
      * https://www.youtube.com/watch?v=5-2C4eO4cPQ 
      * https://www.youtube.com/watch?v=Mc6sBAUdDP4
	* [Moving Average Model MA](https://www.youtube.com/watch?v=voryLhxiPzE) 
    * [Autoregressive Integrated Moving Average Model ARIMA](https://www.youtube.com/watch?v=3UmyHed0iYE)
    * [Seasonality](https://www.youtube.com/watch?v=4hrMdu9CSQs)
    * SARIMA (Seasonal ARIMA)
      * https://www.youtube.com/watch?v=WjeGUs6mzXg
      * https://365datascience.com/tutorials/python-tutorials/arimax/
      * https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
      * https://machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/
      * https://www.alldatascience.com/time-series/forecasting-time-series-with-auto-arima/
      * https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/

    
#### Deep-learning methods

* [**Recurrenct neural networks (RNNs)**](https://www.youtube.com/watch?v=AsNTP8Kwu80)
    * [Long short-term memory (LSTM)](https://www.youtube.com/watch?v=YCzL96nL7j0)

* **Sequence-to-sequence model**
    * Split feature extraction and forecast.
    * Encoder compresses information into hidden state.
    * Decoder uses hidden state to make forecast.

* DeepAR
  * https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/deepar.html
  * https://doi.org/10.1016/j.ijforecast.2019.07.001

* **CNN-LSTM (ENTSO-E dual model)**
    * A Deep Neural Network Model for Short-Term Load Forecast Based on Long Short-Term Memory Network and Convolutional Neural Network [https://doi.org/10.3390/en11123493](https://doi.org/10.3390/en11123493) 
      (implementation for univariate input data (i.e., scalar load), replace 1d convolutions with higher-d convolutions to adapt for multivariate input data)

* **Transformers**

#### Notes
- Include weather data from Deutsche Wetterdienst (DWD), dwd.de, Python interface exists, data probably available as a grid, use averages over all Germany as input features?
- Calculate merit order number for training data and predict that?
- Check flow tracing / production- vs. consumption-based data ENTSO-E.
- Modifying input artificially is a bad idea, Out-Of-Sample Distribution, if one were to do this, NN needs to be trained on synthetic data as well!
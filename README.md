# AI-based Prediction of the Merit Order (of the Marginal Power Plant) to Substitute $\mathbf{CO_2}$ Emissions in the German Electricity Mix

## Hypothesis and goals
If it can be predicted which marginal power plant (technology) determines the merit order (supply and demand curve of the national electricity markets), the amount of $\mathrm{CO_2}$ emissions saved by the additional feed-in of a distributed (renewable) generation plant (merit order effect) can be estimated.
An AI model learns from historical and forecasted national consumption, generation, and weather data as well as cross-border flows which generation technology of which nation provides the marginal power plant under which external conditions and consumption patterns. 
Which technologies contributed when and in which quantity to the national energy demand can be determined by the so-called "flow tracing". 
Based on a graph based representation of the energy system under consideration, flow tracing can be used to determine which technologies provided which amount of energy at each node at any given time. 
This data will then be used as target feature for the prediction model of the merit order. 
If the load or the feed-in of renewable energy sources was changed artificially, the merit order also changes and it becomes visible which technology formed the marginal power plant or generally in the optimization of operating strategies.
The objective of this work is to develop an AI-based predictive model to determine the substituted amount of $\mathrm{CO_2}$ emissions as an operational decision in power plant management.

### Goals:
- Development of a forecast model for the prediction of the cross-market merit order of Germany
- Publication of the results/approaches in a joint scientific publication
- Reusable source code (open source) (non-productive code) without license restrictions

## Tasks
1. Implement dataloader (MW).
    - Use entsoe-py Python package: [https://github.com/EnergieID/entsoe-py](https://github.com/EnergieID/entsoe-py)
    - Use pandas dataframes.
2. Implement persistence (MW).
3. Implement PSLPs (MW).
4. Implement simplest DL model (MW + JST).

## Notes
* What is the actual problem? Multi output regression!
* We are interested in the short-term load forecasting regime (from 24 h to one week). STLF is essential for controlling and scheduling of the power system in making everyday power system operation, interchange evaluation, security assessment, reliability analysis, and spot price calculation, which leads to the higher accuracy requirement compared to long-term prediction.
* We want to predict (time) series of vectors with the fractions of each generation technology with respect to the German electricity mix ("Actual Generation per Production Type", [https://transparency.entsoe.eu/](https://transparency.entsoe.eu/)) from former (time) series. Historic and forecast window are hyperparameters!

### Data
* Which data do we want to use? How does the actual input data look?
* Data is published daily with a resolution of 15 min.
* Each sample is a matrix with the fractions of each generation technology with respect to the German electricty mix for a defined series of consecutive points in time.
* Transforms: Standardization, residuals
* Iterator: Windows <-- historic | forecast -->; naive / stride tricks

#### Actual Generation per Production Type
Actual aggregated net generation output (MW) per market time unit and per production type.
The information shall be published no later than one hour after the operational period.

**Specification of calculation** Average of all available instantaneous net generation output values on each market time unit. If a net generation output is not known, it shall be estimated. The actual generation of small-scale units might be estimated if no real-time measurement devices exist.

**Primary owner of the data** Owners of generation units or transmission system operators (TSOs)  
**Data provider** TSOs or other Data Provider of information depending on local organisation.

#### Actual wind and solar power generation
Actual or estimated wind and solar power net generation (MW) in each bidding zone per market time unit.
A bidding zone is the largest geographical area within which market participants are able to exchange energy without capacity allocation. Within each bidding zone, a single (wholesale) electricity market price applies. Currently, bidding zones in Europe are mostly defined by national borders. 
The information shall to be published no later than one hour after the end of each operating period (of one market time unit length) and be updated on the basis of measured values as soon as they become available. The information shall be provided for all bidding zones only in Member States with more than 1% feed-in of wind or solar power generation per year or for bidding zones with more than 5% feed-in of wind or solar power generation per year.

**Specification of calculation** Average of all available instantaneous power output values on each market time unit. If net power generation output is not known, it shall be estimated.The actual generation of small-scale units might be estimated if no real-time measurement devices exist.

**Primary owner of the data** Owners of generating units and / or distribution system operators (DSOs)  
**Data provider** TSOs or other Data Provider of information depending on local organisation.  
**Aggregation** Locally ( in Data provider)

**Publication deadline for ENTSO-E** H+1 following the concerned MTU  
**Updates** Multiple update possible based on measured data

#### Manual inspection of exemplary data for GER downloaded as CSV file 

There are 23 columns (= raw features):
- A. Area: All the same because we are only interested in Germany (DE).
- B. MTU: Considered time window (start + end with delta = 15 min)
- C. Biomass - Actual Aggregated [MW]
- D. Fossil Brown coal/Lignite - Actual Aggregated [MW]
- E. Fossil Coal-derived gas - Actual Aggregated [MW]
- F. Fossil Gas - Actual Aggregated [MW]
- G. Fossil Hard coal - Actual Aggregated [MW]
- H. Fossil Oil - Actual Aggregated [MW]
- I. Fossil Oil shale - Actual Aggregated [MW]
- J. Fossil Peat - Actual Aggregated [MW]
- K. Geothermal - Actual Aggregated [MW]
- L. Hydro Pumped Storage - Actual Aggregated [MW]
- *M. Hydro Pumped Storage  - Actual Consumption [MW]* WHAT IS THIS?
- N. Hydro Run-of-river and poundage - Actual Aggregated [MW]
- O. Hydro Water Reservoir - Actual Aggregated [MW]
- P. Marine - Actual Aggregated [MW]
- Q. Nuclear - Actual Aggregated [MW]
- R. Other - Actual Aggregated [MW]
- S. Other renewable - Actual Aggregated [MW]
- T. Solar - Actual Aggregated [MW]
- U. Waste - Actual Aggregated [MW]
- V. Wind Offshore - Actual Aggregated [MW]
- W. Wind Onshore - Actual Aggregated [MW]

Do all these generation types appear separately in the merit order or can we combine some of them?  
For our model, we only use the data for GER itself (even though it is also influenced by the data from other countries)?  
How to include the generation forecasts (day ahead, wind + solar)?

#### What do we need to have in our input data?

Raw features:
- time slot (market time unit)
    - Which format? 
- generation technology (production type)
    - One-hot encoded? 
- actual aggregated net generation output in MW (i.e., per market time unit and per production type)
    - Normalize?
    - Calculate residuals w.r.t. PSLP?

Derived features:
- PSLP category (i.e., weekday, Saturday, Sunday / holiday)
    - One-hot-encoded 

### Models

[https://www.entsoe.eu/Technopedia/techsheets/enhanced-load-forecasting](https://www.entsoe.eu/Technopedia/techsheets/enhanced-load-forecasting)
#### Statistical methods

* **Persistence**
    * Used as a baseline model / reference for evaluating a model's predictive power / performance. 
    * Useful to know if a forecast model provides better results than a trivial reference model, here the persistence model.
    * Persistence represents probably the simplest way of producing a forecast. A persistence model assumes that the future value of a time series is calculated under the assumption that nothing changes between the current time and the forecast time. In terms of the electricity mix, the persistence model estimates that the mix at time t+1 equals the mix at time t.

* **Standard load profiles (SLPs)**
    * https://ieeexplore.ieee.org/document/9221967
    * different categories: weekday, Saturday, sun-/holiday
    * Use category-based average of last n profiles as forecast: Last(n) category-forecast
    * Usually: n = 3 (can also capture seasonal + weather related changes)
    * Last(1) = persistence (tomorrow will be the same as today), simple + efficient, but struggles with weekends + holidays.
    * Useful Python package: `holidays`

* **ARIMA (Auto-Regressive Integrated Moving Average) model**
    * SARIMA (Seasonal ARIMA)
    * [https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
    * [https://machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/](https://machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/)
    
#### Deep-learning methods

* **Recurrenct neural networks (RNNs)**
    * Long short-term memory (LSTM)

* **Sequence-to-sequence model**
    * Split feature extraction and forecast.
    * Encoder compresses information into hidden state.
    * Decoder uses hidden state to make forecast.

* **CNN-LSTM (ENTSO-E dual model)**
    * A Deep Neural Network Model for Short-Term Load Forecast Based on Long Short-Term Memory Network and Convolutional Neural Network [https://doi.org/10.3390/en11123493](https://doi.org/10.3390/en11123493) (implementation for univariate input data (i.e., scalar load), replace 1d convolutions with higher-d convolutions to adapt for multivariate input data)

* **Transformers**

### Meeting minutes
#### 2022-12-12
- Implemented minimal examples for querying data from ENTSO-E transparency platform via Python interface for RESTful API with `entsoe-py` and `entsoe-client` ([https://pypi.org/project/entsoe-client/](https://pypi.org/project/entsoe-client/)). See Jupyter notebooks.
- Error for `entsoe-py` => Github issue [https://github.com/EnergieID/entsoe-py/issues/225](https://github.com/EnergieID/entsoe-py/issues/225), loading only works for Belgium but not for Germany!
- Loading with `entsoe-client` works but is kind of hard to follow.

- **TODO** Understand data loading with `entsoe-client` in detail, in particular handling of NaN.
- **TODO** Wait for response to issue.
- **TODO** How to passe pandas dataframe to PyTorch dataloader?

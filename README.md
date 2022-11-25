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
2. Implement persistence (MW).
3. Implement PSLPs (MW).
4. Implement simplest DL model (MW + JST).

## Notes
* What is the actual problem? Multi output regression!
* We are interested in the short-term load forecasting regime (from 24 h to one week). STLF is essential for controlling and scheduling of the power system in making everyday power system operation, interchange evaluation, security assessment, reliability analysis, and spot price calculation, which leads to the higher accuracy requirement compared to long-term prediction.
* We want to predict (time) series of vectors with the fractions of each generation technology with respect to the German electricity mix from former (time) series. Historic and forecast window are hyperparameters!

### Data
* Which data do we want to use? How does the actual input data look?
* Each sample is a vector with the fractions of each generation technology with respect to the German electricty mix at a certain point in time
* What is the temporal resolution of the data? Hourly? Time step t_i+1 = t_i + delta t with delta t = 24 h?
* Transforms: Standardization, residuals
* Iterator: Windows <-- historic | forecast -->; naive / stride tricks

### Models

https://www.entsoe.eu/Technopedia/techsheets/enhanced-load-forecasting
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
    * Useful Python package: holidays

* **ARIMA (Auto-Regressive Integrated Moving Average) model**

#### Deep-learning methods

* **Recurrenct neural networks (RNNs)**
    * Long short-term memory (LSTM)

* **Sequence-to-sequence model**
    * Split feature extraction and forecast.
    * Encoder compresses information into hidden state.
    * Decoder uses hidden state to make forecast.

* **CNN-LSTM (ENTSO-E dual model)**
    * A Deep Neural Network Model for Short-Term Load Forecast Based on Long Short-Term Memory Network and Convolutional Neural Network https://doi.org/10.3390/en11123493 (implementation for univariate input data (i.e., scalar load), replace 1d convolutions with higher-d convolutions to adapt for multivariate input data)

* **Transformers**

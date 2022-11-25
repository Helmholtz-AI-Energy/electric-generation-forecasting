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

from models.PSLP import PersonalizedStandardizedLoadProfile
from models.plot_results import plot_prediction
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def modelling_pslp(data, previous_days, forecast_horizon, scores, method, sector, mode, plot_opt):
    pslp = PersonalizedStandardizedLoadProfile(target_column='load', data_frequency='15Min',
                                               forecast_horizon=f'{forecast_horizon}D', country='DE', state='NI')

    pred = pd.DataFrame()
    real = pd.DataFrame()
    load_data = data['load']
    for x in tqdm(range(previous_days, int(len(load_data) / 96) - forecast_horizon - 1)):
        i = x * 96
        if x == previous_days:
            current_data = load_data.iloc[:i].to_frame()
        else:
            current_data = load_data.iloc[i - 96:i].to_frame()
        pslp.preprocess_new_data(current_data)

        time_now = load_data.iloc[i - 96:i+1].last_valid_index()
        prediction = pslp.forecast_standard(time_now)
        pred = pd.concat([pred, prediction], axis=0)
        real_measurements = load_data[i:i + forecast_horizon * 96]
        real = pd.concat([real, real_measurements], axis=0)

        mae = mean_absolute_error(prediction, real_measurements)
        mse = mean_squared_error(prediction, real_measurements)
        mape = mean_absolute_percentage_error(prediction, real_measurements)
        scores.loc[time_now, f'mae_pslp_std'] = mae
        scores.loc[time_now, f'mse_pslp_std'] = mse
        scores.loc[time_now, f'mape_pslp_std'] = mape * 100
        if plot_opt == True:
            plot_prediction(prediction, real_measurements, time_now, method, mode, sector)

    return scores, pred, real


def modelling_pslp_fix(data, previous_days, forecast_horizon, scores, method, sector, mode, plot_opt):
    pslp = PersonalizedStandardizedLoadProfile(target_column='load', forecast_horizon=f'{forecast_horizon}D',
                                               data_frequency='15Min', country='DE', state='NI')
    pslp.set_up_fixed_profile_look_back(2, 7, 7, 2, 7, 7, 2, 7, 7)
    load_data = data['load']
    pred = pd.DataFrame()
    real = pd.DataFrame()
    for x in tqdm(range(previous_days, int(len(load_data) / 96) - forecast_horizon - 1)):
        i = x * 96
        if x == previous_days:
            current_data = load_data.iloc[:i].to_frame()
        else:
            current_data = load_data.iloc[i - 96:i].to_frame()
        pslp.preprocess_new_data(current_data)
        time_now = load_data.iloc[i - 96:i].last_valid_index()
        prediction = pslp.forecast_fixed_profile_look_back(time_now)
        pred = pd.concat([pred, prediction], axis=0)

        real_measurements = load_data[i:i + forecast_horizon * 96]
        real = pd.concat([real, real_measurements], axis=0)

        mae = mean_absolute_error(prediction, real_measurements)
        mse = mean_squared_error(prediction, real_measurements)
        mape = mean_absolute_percentage_error(prediction, real_measurements)
        scores.loc[time_now, f'mae_pslp_fix'] = mae
        scores.loc[time_now, f'mse_pslp_fix'] = mse
        scores.loc[time_now, f'mape_pslp_fix'] = mape * 100
        if plot_opt == True:
            plot_prediction(prediction, real_measurements, time_now, method, mode, sector)

    return scores, pred, real


def modelling_pslp_var(data, previous_days, forecast_horizon, scores, method, sector, mode, plot_opt):
    pslp = PersonalizedStandardizedLoadProfile(target_column='load', forecast_horizon=f'{forecast_horizon}D',
                                               data_frequency='15Min', seasonal=False, country='DE', state='NI')
    pslp.set_up_variable_profile_look_back(21)
    pred = pd.DataFrame()

    load_data = data['load']
    saved_predictions = pd.DataFrame()
    real = pd.DataFrame()
    for x in tqdm(range(previous_days, int(len(load_data) / 96) - forecast_horizon - 1)):
        i = x * 96
        if x == previous_days:
            current_data = load_data.iloc[:i].to_frame()
        else:
            current_data = load_data.iloc[i - 96:i].to_frame()
        historical_data = load_data.iloc[i - 96 * 7:i].to_frame()
        pslp.preprocess_new_data(current_data)
        prediction_data = load_data[i - previous_days * 96:i - previous_days * 96 + forecast_horizon * 96]
        time_now = load_data.iloc[i - 96:i].last_valid_index()
        prediction = pslp.forecast_variable_profile_look_back(time_now, historical_data)
        real_measurements = load_data[i:i + forecast_horizon * 96]
        # plt.plot(real_measurements.values, label ='real')
        # plt.plot(prediction.values, label='pred')
        # plt.legend()
        # plt.show()
        if prediction.shape[0] > 192:
            prediction = prediction[:-1]
        mae = mean_absolute_error(prediction, real_measurements)
        mse = mean_squared_error(prediction, real_measurements)
        mape = mean_absolute_percentage_error(prediction, real_measurements)
        scores.loc[time_now, f'mae_pslp_var'] = mae
        scores.loc[time_now, f'mse_pslp_var'] = mse
        scores.loc[time_now, f'mape_pslp_var'] = mape * 100
        # plt.plot(prediction_data.reset_index(drop=True))
        # plt.plot(real_measurements.reset_index(drop=True))
        # plt.show()
        pred = pd.concat([pred, prediction], axis=0)
        real = pd.concat([real, real_measurements], axis=0)

        if method == "ml":
            prediction[x] = prediction['forecast']
            saved_predictions = pd.concat([saved_predictions, prediction[x].reset_index(drop=True)], axis=1)
        #prediction[x] = prediction['forecast']
        else:
        #saved_predictions = pd.concat([saved_predictions, prediction[x].reset_index(drop=True)], axis=1)
            if plot_opt == True:
                plot_prediction(prediction, real_measurements, time_now, method, mode, sector)

    if method == "ml":
        return saved_predictions, scores
    else:
        return scores, pred, real


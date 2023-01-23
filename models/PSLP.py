import holidays
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error


class PersonalizedStandardizedLoadProfile:
    """
    What is it?
    -----------
        Personalized Standardized Load Profiles (short PSLP) can be seen as a successor of commonly used standardized laod
        profiles, derived by the VDEW (now BDEW) in 2000 (https://www.bdew.de/energie/standardlastprofile-strom/). With the
        transition to smart meter data gets widely available and with that possibility to use these data to build load
        profiles for single buildings arises. Based on Hinterstocker et Al. (BEWERTUNG DER AKTUELLEN STANDARDLASTPROFILE
        ÖSTERREICHS UND ANALYSE ZUKÜNFTIGER ANPASSUNGSMÖGLICHKEITEN IM STROMMARKT) the ruleset derived by the VDEW was
        adopted to form profiles for different regions. This idea was then adopted during DLR project EMGIMO for single
        building.

        This class aims to propose different methods to forecast future values based on the VDEW ideas and ruleset but with
        the extension of further knowledge gained for example from EMGIMO and EMMMS as a follow up project.


    Principle
    ---------
        In general this class has the option to do seasonal and day type  classification of data. These can be switched
        on and off based on the time series in question. E.G. for a commercial building where nobody is working on the
        weekend and which has a somehow yearly pattern. General public vacations can be used aswell and are always treated
        as a sunday (in coherence to general SLP Rules)

        PSLP (Standard)
        With seasonal and day type enables this is the standard method. It collects data and puts it into 9 different
        profiles (3 Seasons: Transition, Summer, Winter; and 3 day types: Weekday, Saturday and Sunday/Vacation). New data
        is then classified and all data within one profile are averaged to calculate the profile to be applied.

        PSLP Fixed
        In difference to standard PSLP the usage of data to form a profile is limited to last n-days. It has the advantage
        that on longer periods (e.g. some years) old data is not carried through all the time and can be "forgotten". This
        makes the profiles more flexible than using  standard PSLP

        PSLP variable
        This is the most advanced PSLP. Like the fixed PSLP the look_back is limited but the time span is assigned
        dynamically based on a given metric. This can for example be based on MAPE, MAE ...



    Attributes
    ----------

        target_column : str
            name of target column in a given pandas dataframe

        forecast_horizon : str
            duration to predict into the future given as str (e.g. '2D' for 2 days)

        data_frequency : str
            frequency which incoming data has given as str (e.g. "5Min" for 5 minutely data)

        seasonal : bool
            seasonal seperation for profiles

        day_type : bool
            seperation of data into: weekday, saturday, sunday/vacation

        country : str
            country code from holidays package

        state : str
            state code from holidays if country is none state has no effect

        aggregation_method : str
            aggregation method to use, can be set to mean or median

        metric : str
            metric to evaluate variable PSLP (mean_absolute_error, mean_squared_error or mean_absolute_percentage_error)

        profile_storage : dict
            stores classified data in selected profiles

        steps_per_day : int
            steps on one day (e.g. 15 Minutes resolution -> 96 Steps on one day)

        profile_look_back : dict or None
            stores used look_backs for different profiles

        maximum_look_back : int
            maximum of days to look_back

        profile_look_back_set : dict
            stores information on which look_back is already set

        test_pred_len : int
            test prediction length during look back evaluation



    Methods
    -------
        preprocess_new_data(incoming_data)
            Sort data into profiles to be used during prediction

        forecast_standard(time_now)
            forecast with standard PSLP method (options set for season and day type apply!)

        set_up_fixed_profile_look_back(ww=7, wsa=7, wsu=7, sw=7, ssa=7, ssu=7, tw=7, tsa=7, tsu=7)
            sets look_back for every profile to be used

        forecast_fixed_profile_look_back(time_now)
            forecast with fixed look_back PSLP method (options set for season and day type apply!)

        set_up_variable_profile_look_back(maximum_look_back):
            sets up variable look back pslp and sets maximum look_back to use

        forecast_variable_profile_look_back(time_now, historical_data):
            forecast with variable look_back PSLP method (options set for season and day type apply!)
    
    """

    def __init__(self, target_column: str, forecast_horizon: str = '2D', data_frequency: str = '5Min', seasonal=True,
                 day_type=True, country=None, state=None, aggregation_method='mean', metric='mean_squared_error', test_pred_len=1):
        """

        Parameters
        ----------
            target_column : str
                name of target column in a given pandas dataframe

            forecast_horizon : str
                duration to predict into the future given as str (e.g. '2D' for 2 days)

            data_frequency : str
                frequency which incoming data has given as str (e.g. "5Min" for 5 minutely data)

            seasonal : bool
                seasonal seperation for profiles

            day_type : bool
                seperation of data into: weekday, saturday, sunday/vacation

            country : str
                country code from holidays package

            state : str
                state code from holidays if country is none state has no effect

            aggregation_method : str
                aggregation method to use, can be set to mean or median

            metric : str
                metric to evaluate variable PSLP (mean_absolute_error, mean_squared_error or mean_absolute_percentage_error)

        
        """
        self.test_pred_len = test_pred_len
        self.target_column = target_column
        self.forecast_horizon = forecast_horizon
        self.data_frequency = data_frequency
        self.seasonal = seasonal
        self.day_type = day_type
        self.country = country
        self.state = state
        self.aggregation_method = aggregation_method
        self.metric = metric
        self.profile_storage = None
        self.steps_per_day = int(pd.to_timedelta('24H') / pd.to_timedelta(self.data_frequency))
        self.profile_look_back = None
        self.maximum_look_back = None
        self.profile_look_back_set = None

        self.__init_profile_storages()

    def __init_profile_storages(self):
        """
        initializes data storages as a dict of dataframes to store all data in which are needed to calculate different
        profiles

        Parameter
        ---------
            None

        Return
        ------
            None
        """
        time_index = pd.date_range('00:00:00', '23:59:59', freq=self.data_frequency).time
        self.profile_storage = dict()
        if self.day_type is True and self.seasonal is True:
            day_types_to_compute = ['ww', 'wsa', 'wsu', 'tw', 'tsa', 'tsu', 'sw', 'ssa', 'ssu']
        elif self.day_type is False and self.seasonal is True:
            day_types_to_compute = ['ww', 'tw', 'sw']
        elif self.day_type is True and self.seasonal is False:
            day_types_to_compute = ['sw', 'ssa', 'ssu']
        else:
            day_types_to_compute = ['sw']
        for element in day_types_to_compute:
            self.profile_storage[element] = pd.DataFrame(index=time_index)

    def __init_profile_look_back_test_dicts(self):
        """
        Inits dict where the initial value for all profiles is 0

        Parameter
        ---------
            None

        Return
        ------
            None
        """
        dict_with_empty_profiles = dict()
        if self.day_type is True and self.seasonal is True:
            day_types_to_compute = ['ww', 'wsa', 'wsu', 'tw', 'tsa', 'tsu', 'sw', 'ssa', 'ssu']
        elif self.day_type is False and self.seasonal is True:
            day_types_to_compute = ['ww', 'tw', 'su']
        elif self.day_type is True and self.seasonal is False:
            day_types_to_compute = ['sw', 'ssa', 'ssu']
        else:
            day_types_to_compute = ['sw']
        for element in day_types_to_compute:
            dict_with_empty_profiles[element] = 0  # init all day_type profiles
        return dict_with_empty_profiles

    def _vacation_dates(self, year):
        """
        Retrieves dates on which vacations are in a specific country (optional. state)

        Parameter
        ---------
            year : str
                year to calculate vacations for

        Return
        ------
            de_holidays : list
                list of holidays within the year
        """
        if self.country is None:
            return []
        else:
            country = self.country
            state = self.state
            de_holidays = list(holidays.CountryHoliday(country, prov=state, years=[year]).keys())
            return de_holidays

    def _get_season_info(self, date, year):
        """
        Gets seasonal information of date (if option is activated), otherwise it is set always to summer (s)

        Parameter
        ---------
            date : str
                date which should be classified
            year : str
                year in which the date is

        Return
        ------
            season in which the date is
        """
        if self.seasonal is False:
            return 's'
        elif self.seasonal is True:
            if pd.to_datetime(year + '-01-01').date() <= date <= pd.to_datetime(
                    year + '-03-20').date() or pd.to_datetime(year + '-11-01').date() <= date <= pd.to_datetime(
                    year + '-12-31').date():
                season = 'w'  # winter
            elif pd.to_datetime(year + '-03-21').date() <= date <= pd.to_datetime(
                    year + '-05-14').date() or pd.to_datetime(
                    year + '-09-15').date() <= date <= pd.to_datetime(year + '-10-31').date():
                season = 't'  # transition
            elif pd.to_datetime(year + '-05-15').date() <= date <= pd.to_datetime(
                    year + '-09-14').date():
                season = 's'  # summer
            else:
                raise ValueError('Undefined Date')
            return season

    @staticmethod
    def _check_for_christmas_new_year(year, date, day_of_week):
        """
        checks if date is on christmas day or new years

        Parameter
        ---------
            date : str
                date which should be classified
            year : str
                year in which the date is
            day_of_week : int
                day of week encoded as an int value

        Return
        ------
            corrected day type or None
        """
        if pd.to_datetime(year + '-12-24 00:00').date() == date or date == pd.to_datetime(year + '-12-31 00:00').date():
            if day_of_week == 6:
                return 'su'
            else:
                return 'sa'
        else:
            return None

    def _check_if_vacation(self, date):
        """
        checks if date is on public holiday

        Parameter
        ---------
            date : str
                date which should be classified

        Return
        ------
            Bool
        """
        # check if date is a vacation date
        if date in self._vacation_dates(date.year):
            return True
        else:
            return False

    def _get_day_type(self, year, date):
        """
        checks which day type current date is (if option is enabled), otherwise it is always set to week (w)

        Parameter
        ---------
            date : str
                date which should be classified
            year : str
                year in which the date is

        Return
        ------
            day_type : str
        """
        day_of_week = pd.to_datetime(date).dayofweek
        christmas_new_years_day_type = self._check_for_christmas_new_year(year, date, day_of_week)
        vacation = self._check_if_vacation(date)
        if self.day_type is False:
            return 'w'
        if christmas_new_years_day_type is None:
            if day_of_week <= 4 and vacation is False:
                return 'w'
            elif day_of_week <= 4 and vacation is True:
                return 'su'
            elif day_of_week == 5 and vacation is False:
                return 'sa'
            elif day_of_week == 5 and vacation is True:
                return 'su'
            elif day_of_week == 6:
                return 'su'
            else:
                print('oops')
        else:
            return christmas_new_years_day_type

    def _classification_day(self, date):
        """
        classification of day according to specified ruled by SLP Method and enabled options

        Parameter
        ---------
            date : str
                date which should be classified
            year : str
                year in which the date is

        Return
        ------
            day_type : str
                day
        """
        year = str(date.year)
        season = self._get_season_info(date, year)
        day_type = self._get_day_type(year, date)
        return season, day_type

    def _sort_data_into_profile_data_storage(self, profile_name, data_on_date, date):
        """
        sorts incoming data into different profile data caches

        Parameter
        ---------
            date : str
                date which should be classified
            year : str
                year in which the date is

        Return
        ------
            day_type : str
                day
        """
        stored_measurements_profile = self.profile_storage.get(profile_name)
        data_on_date.index = data_on_date['time']
        stored_measurements_profile.loc[data_on_date.index, date] = data_on_date[self.target_column]
        self.profile_storage[profile_name] = stored_measurements_profile.copy()

    def preprocess_new_data(self, incoming_data):
        """
        preprocesses new data and sort data into profiles

        Parameter
        ---------
            incoming_data : pd.DataFrame
                DataFrame with data of the recent timestep (e.g. if 24H are gone since it was last time called
                this should have all data from the last 24H in it)

        Return
        ------
            None

        """
        data_to_preprocess = incoming_data.copy(deep=True)
        data_to_preprocess['date'] = pd.to_datetime(data_to_preprocess.index).date
        data_to_preprocess['time'] = pd.to_datetime(data_to_preprocess.index).time
        unique_dates = data_to_preprocess['date'].unique()
        for date in unique_dates:
            data_on_date = data_to_preprocess.loc[data_to_preprocess['date'] == date]
            season, day_info = self._classification_day(date)
            profile_name = season + day_info
            self._sort_data_into_profile_data_storage(profile_name, data_on_date, date)

    def _round_up(self, time_now):
        """
        Rounds up time to next full frequency time (e.g. 16:56:59 -> 17:00:00)

        Parameter
        ---------
            time_now : pd.datetime
                current time to be rounded

        Return
        ------
            rounded date time
        """
        # find next timestep which is regular (e.g. 16:56:59 -> 17:00:00)
        rounded_time = time_now.round(self.data_frequency)
        # if time gets rounded down -> round it up
        if rounded_time < time_now:
            rounded_time = rounded_time + pd.to_timedelta(self.data_frequency)
        return rounded_time

    @staticmethod
    def _get_previous_profile(season, date):
        """
        Get previous profile by getting previous season

        Parameter
        ---------
            season : str
                season date was sorted to

            date : pd.datetime
                current date


        Return
        ------
            previous season
        """
        # if there is no data in the profile for current season, use values of the last season as a guess
        year = str(date.year)
        if season == 'w' or season == 's':
            return 't'
        elif season == 't':
            if pd.to_datetime(year + '-05-15').date() <= date <= pd.to_datetime(year + '-09-14').date():
                return 's'
            else:
                return 'w'
        else:
            raise ValueError('Previous Season not found!')

    def _aggregate_to_profile(self, stored_measurements_profile):
        """
        calculates profile from preprocessed and to be used data

        Parameter
        ---------
            cached_data : pd.DataFrame
                sorted data to be used for profile

        Return
        ------
            calculated profile
        """
        if self.aggregation_method == 'mean':
            return stored_measurements_profile.mean(axis=1)
        elif self.aggregation_method == 'median':
            return stored_measurements_profile.median(axis=1)
        else:
            raise ValueError('Aggregation has to be set to mean or median')

    def forecast_standard(self, time_now):
        """
        Forecast using standard PSLP-method

        Parameter
        ---------
            time_now : pd.datetime
                current time to start forecasting from

        Return
        ------
            forecast
        """
        time_now_rounded_up = self._round_up(time_now)
        forecast_index = pd.date_range(start=time_now_rounded_up,
                                       end=pd.to_datetime(time_now_rounded_up) + pd.to_timedelta(
                                           self.forecast_horizon) - pd.to_timedelta(self.data_frequency),
                                       freq=self.data_frequency)
        forecast_data = pd.DataFrame(index=forecast_index)
        forecast_data['time'] = forecast_data.index.time
        forecast_data['date'] = forecast_data.index.date
        unique_dates = np.unique(forecast_data.index.date)
        for date in unique_dates:
            season, day_info = self._classification_day(date)
            name_of_profile = season + day_info
            cached_data = self.profile_storage.get(name_of_profile).fillna(method="ffill")
            profile = self._aggregate_to_profile(cached_data)
            # if the profile derived is empty or has not all times get previous profile
            if profile.empty or profile.dropna().shape[0] < self.steps_per_day:
                season = self._get_previous_profile(season, date)
                name_of_profile = season + day_info
                cached_data = self.profile_storage.get(name_of_profile)
                profile = self._aggregate_to_profile(cached_data)
            data_for_day = forecast_data.loc[forecast_data['date'] == date].copy()
            data_for_day.loc[:, 'index'] = data_for_day.index
            data_for_day.index = data_for_day['time']
            data_for_day.loc[:, 'forecast'] = profile.loc[data_for_day.index]
            data_for_day.index = data_for_day['index']
            forecast_data.loc[data_for_day.index, 'forecast'] = data_for_day['forecast']
        return forecast_data.drop(columns=['time', 'date'])

    def set_up_fixed_profile_look_back(self, ww=7, wsa=7, wsu=7, sw=7, ssa=7, ssu=7, tw=7, tsa=7, tsu=7):
        """
        Setup a dictionary containing chosen look_backs

        Parameter
        ---------
            ww : int
                look_back for profile winter week

            wsa : int
                look_back for profile winter saturday

            wsu : int
                look_back for profile winter sunday/vacation

            sw : int
                look_back for profile summer week

            ssa : int
                look_back for profile summer saturday

            ssu : int
                look_back for profile summer sunday/vacation

            tw : int
                look_back for profile transition week

            tsa : int
                look_back for profile transition saturday

            tsu : int
                look_back for profile transition sunday/vacation

        Return
        ------
            None
        """
        self.profile_look_back = {'ww': ww, 'wsa': wsa, 'wsu': wsu, 'sw': sw, 'ssa': ssa, 'ssu': ssu, 'tw': tw,
                                  'tsa': tsa,
                                  'tsu': tsu}

    def forecast_fixed_profile_look_back(self, time_now):
        """
        Forecast using fixed look_back PSLP-method

        Parameter
        ---------
            time_now : pd.datetime
                current time to start forecasting from

        Return
        ------
            forecast
        """
        if self.profile_look_back is None:
            raise ValueError('Specify profile_look_back by using set_up_fixed_profile_look_back')
        time_now_rounded_up = self._round_up(time_now)
        forecast_index = pd.date_range(start=time_now_rounded_up,
                                       end=pd.to_datetime(time_now_rounded_up) + pd.to_timedelta(
                                           self.forecast_horizon) - pd.to_timedelta(self.data_frequency),
                                       freq=self.data_frequency)
        forecast_data = pd.DataFrame(index=forecast_index)
        forecast_data['time'] = forecast_data.index.time
        forecast_data['date'] = forecast_data.index.date
        unique_dates = np.unique(forecast_data.index.date)
        for date in unique_dates:
            season, day_info = self._classification_day(date)
            name_of_profile = season + day_info
            cached_data = self.profile_storage.get(name_of_profile)
            used_profile_look_back = self.profile_look_back.get(name_of_profile)
            profile = self._aggregate_to_profile(cached_data.iloc[:, -used_profile_look_back:])
            # if the profile derived is empty or has not all times get previous profile
            if profile.empty or profile.dropna().shape[0] < self.steps_per_day:
                season = self._get_previous_profile(season, date)
                name_of_profile = season + day_info
                cached_data = self.profile_storage.get(name_of_profile)
                profile = self._aggregate_to_profile(cached_data.iloc[:, -used_profile_look_back:])
            data_for_day = forecast_data.loc[forecast_data['date'] == date].copy()
            data_for_day.loc[:, 'index'] = data_for_day.index
            data_for_day.index = data_for_day['time']
            data_for_day.loc[:, 'forecast'] = profile.loc[data_for_day.index]
            data_for_day.index = data_for_day['index']
            forecast_data.loc[data_for_day.index, 'forecast'] = data_for_day['forecast']
        return forecast_data.drop(columns=['time', 'date'])

    def _profile_look_back_reset(self):
        """
        resets all states used to track which profile already has a set look back

        Parameter
        ---------
            None

        Return
        ------
            None
        """
        self.profile_look_back_set = {'ww': False, 'wsa': False, 'wsu': False, 'sw': False, 'ssa': False, 'ssu': False,
                                      'tw': False, 'tsa': False, 'tsu': False}

    def set_up_variable_profile_look_back(self, maximum_look_back):
        """
        sets up variable PSLP

        Parameter
        ---------
            maximum_look_back : int
                maximum value to test for look_back

        Return
        ------
            None
        """
        self.set_up_fixed_profile_look_back()
        self.maximum_look_back = maximum_look_back

    def _check_dates_in_data(self, historical_data):
        """
        determine day types and seasons present in historical dataset

        Parameter
        ---------
            historical_data : pd.DataFrame
                data with historical values

        Return
        ------
            day_types_in_hist_data : list
                day types present in historical dataset

            copy_hist_data : pd.DataFrame
                copy of the gistorical data containing a column with day type
        """
        copy_hist_data = historical_data.copy()
        historical_data['time'] = historical_data.index.time
        historical_data['date'] = historical_data.index.date
        unique_dates = np.unique(historical_data.index.date)
        day_types_in_hist_data = self.__init_profile_look_back_test_dicts()
        copy_hist_data['day_type'] = copy_hist_data.index.date
        for date in unique_dates:
            season, day_info = self._classification_day(date)
            name_of_profile = season + day_info
            day_types_in_hist_data[name_of_profile] = day_types_in_hist_data.get(name_of_profile) + 1
            copy_hist_data['day_type'].replace(to_replace=date, value=name_of_profile, inplace=True)
        return day_types_in_hist_data, copy_hist_data

    def _preset_unused_day_types(self, day_type_dataset, name_of_profile):
        """
        presets look back for unused day types

        Parameter
        ---------
            day_type_dataset : pd.DataFrame
                dataset with day types

            name_of_profile : str
                day type (incl. season)

        Return
        ------
            recencies_to_set : list
                which look back for which profile has to be set

        """
        recencies_to_set = []
        count = day_type_dataset.get(name_of_profile)
        sorted_data = self.profile_storage.get(name_of_profile).shape[1]
        if not count >= 1 and not sorted_data >= 1:
            self.profile_look_back[name_of_profile] = 7
        else:
            recencies_to_set.append(name_of_profile)
        return recencies_to_set

    @staticmethod
    def __test_prediction(hist_data, profile):
        """
        predicts using a given profile

        Parameter
        ---------
            hist_data : pd.DataFrame
                historical data on which it is tested

            profile : pd.DataFrame
                profile to be used

        Return
        ------
            hist_data : pd.DataFrame
                prediction

        """
        hist_data['prediction'] = hist_data.index.time
        dictionary = profile.to_dict()
        hist_data.replace({'prediction': dictionary}, inplace=True)
        return hist_data

    def _evaluate(self, prediction):
        """
        evaluates the prediction based on given metric

        Parameter
        ---------
            prediction : pd.DataFrame
                prediction to be evaluated including real measurements

            profile : pd.DataFrame
                profile to be used

        Return
        ------
            score
        """
        if self.metric.lower() == 'mse' or self.metric.lower() == 'mean_squared_error':
            return mean_squared_error(prediction[self.target_column], prediction['prediction'])
        elif self.metric.lower() == 'mape' or self.metric.lower() == 'mean_absolute_percentage_error':
            return mean_absolute_percentage_error(prediction[self.target_column], prediction['prediction'])
        elif self.metric.lower() == 'mae' or self.metric.lower() == 'mean_absolute_error':
            return mean_absolute_error(prediction[self.target_column], prediction['prediction'])
        else:
            raise ValueError('Metric not found!')

    def _test_profile_look_back(self, historical_data, name_of_profile):
        """
        evaluates the prediction based on given metric

        Parameter
        ---------
            hist_data : pd.DataFrame
                historical data on which it is tested

            name_of_profile : str
                day type (incl. season)

        Return
        ------
            None
        """
        copy_hist_data = historical_data.copy()
        day_types_in_hist_data, classified_data = self._check_dates_in_data(copy_hist_data)
        recencies_to_set = self._preset_unused_day_types(day_types_in_hist_data, name_of_profile)
        profile_look_back_set = self.profile_look_back_set.get(name_of_profile)
        if profile_look_back_set is False:
            for day_type in recencies_to_set:
                best_profile_look_back = None
                best_error = None
                cached_data = self.profile_storage.get(day_type)
                for profile_look_back in range(2, self.maximum_look_back):
                    profile = self._aggregate_to_profile(cached_data.iloc[:, -profile_look_back:])
                    day_type_hist_data = classified_data[classified_data['day_type'] == day_type].copy()
                    if cached_data.shape[1] < profile_look_back and not day_type_hist_data.empty:
                        break
                    elif cached_data.shape[1] < profile_look_back and day_type_hist_data.empty:
                        best_profile_look_back = 1
                        break
                    elif day_type_hist_data.empty:
                        best_profile_look_back = 1
                        break
                    else:
                        day_type_hist_data = day_type_hist_data.iloc[-self.test_pred_len * self.steps_per_day:, :]
                        prediction = self.__test_prediction(day_type_hist_data, profile)
                        error = self._evaluate(prediction)
                        if best_profile_look_back is None:
                            best_profile_look_back = profile_look_back
                        if best_error is None:
                            best_error = error
                        if best_error > error:
                            best_profile_look_back = profile_look_back
                            best_error = error
                self.profile_look_back[day_type] = best_profile_look_back
                self.profile_look_back_set[name_of_profile] = True
                print(best_profile_look_back, best_error)

    def forecast_variable_profile_look_back(self, time_now, historical_data):
        """
        Forecast using variable look_back PSLP-method including finding best look back

        Parameter
        ---------
            historical_data : pd.DataFrame
                historical data on which it is tested

            time_now : pd.datetime
                current time to start forecasting from

        Return
        ------
            prediction
        """
        if self.maximum_look_back is None:
            raise ValueError('Set testing length by using set_up_variable_profile_look_back method')
        time_now_rounded_up = self._round_up(time_now)
        forecast_index = pd.date_range(start=time_now_rounded_up,
                                       end=pd.to_datetime(time_now_rounded_up) + pd.to_timedelta(
                                           self.forecast_horizon) - pd.to_timedelta(self.data_frequency),
                                       freq=self.data_frequency)
        forecast_data = pd.DataFrame(index=forecast_index)
        forecast_data['time'] = forecast_data.index.time
        forecast_data['date'] = forecast_data.index.date
        unique_dates = np.unique(forecast_data.index.date)
        number_of_days = int((len(historical_data) / self.steps_per_day)) - 1
        self._profile_look_back_reset()
        for date in unique_dates:
            season, day_info = self._classification_day(date)
            name_of_profile = season + day_info
            if number_of_days < 2:
                profile_look_back = number_of_days
                used_profile_look_back = profile_look_back
            else:
                self._test_profile_look_back(historical_data, name_of_profile)
                used_profile_look_back = self.profile_look_back.get(name_of_profile)

            cached_data = self.profile_storage.get(name_of_profile)
            if used_profile_look_back is not None:
                profile = self._aggregate_to_profile(cached_data.iloc[:, -used_profile_look_back:])
            else:
                season = self._get_previous_profile(season, date)
                name_of_profile = season + day_info
                if number_of_days < 2:
                    profile_look_back = number_of_days
                    used_profile_look_back = profile_look_back
                else:
                    self._test_profile_look_back(historical_data, name_of_profile)
                    used_profile_look_back = self.profile_look_back.get(name_of_profile)
                cached_data = self.profile_storage.get(name_of_profile)
                profile = self._aggregate_to_profile(cached_data.iloc[:, -used_profile_look_back:])
            if profile.empty or profile.dropna().shape[0] < self.steps_per_day:
                season = self._get_previous_profile(season, date)
                name_of_profile = season + day_info
                if number_of_days < 2:
                    profile_look_back = number_of_days
                    used_profile_look_back = profile_look_back
                else:
                    self._test_profile_look_back(historical_data, name_of_profile)
                    used_profile_look_back = self.profile_look_back.get(name_of_profile)
                cached_data = self.profile_storage.get(name_of_profile)
                if used_profile_look_back is None:
                    used_profile_look_back = 1
                    print("test")
                profile = self._aggregate_to_profile(cached_data.iloc[:, -used_profile_look_back:])
            data_for_day = forecast_data.loc[forecast_data['date'] == date].copy()
            data_for_day.loc[:, 'index'] = data_for_day.index
            data_for_day.index = data_for_day['time']
            data_for_day.loc[:, 'forecast'] = profile.loc[data_for_day.index]
            data_for_day.index = data_for_day['index']
            forecast_data.loc[data_for_day.index, 'forecast'] = data_for_day['forecast']
        return forecast_data.drop(columns=['time', 'date'])

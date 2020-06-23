import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa import stattools as sm


class CreateMasterTable:
    def __init__(self, stock, data, volatility, mean_returns, bound_limits):
        """

        :param stock: str, the stock being run
        :param price_data: dataframe, price data for the stock being run
        :param volatility: series, fitted volatility measurements for the stock
        :param start_date: datetime, the start date being run for
        :param end_date: datetime, the end date being run for
        """
        self.data = data
        self.stock = stock
        self.stock_dict = data.stock_dict
        self.price_data = data.stock_dict[stock]
        self.start_date = data.start_date
        self.end_date = data.end_date
        self.volatility = volatility
        self.mean_returns = mean_returns
        self.bound_limits = bound_limits

    def base_table_create(self):
        """
        Create the base tables used to fit models, including labels and features

        :return base_table: a fully labeled (endog and exog variables) date table to be used in our models,
        """

        # Create a base_table with price, and create columns housing base MFI, cointegration, arima fit, arima residuals
        base_table = self.price_data.copy()
        # apply labels to data using triple barrier method
        upper_triggers = np.sqrt(self.volatility)*self.bound_limits
        lower_triggers = -np.sqrt(self.volatility)*self.bound_limits
        base_table['LABEL'], base_table['STRIKE_DATE'] = self.label_observations(base_table, upper_triggers,\
                                                                                 lower_triggers, mean_return=self.mean_returns)
        # truncate the base_table for the dates we are running for (used mainly in benchmark)
        base_table = base_table.loc[(base_table['DATE'] <= self.end_date) & (base_table['DATE'] >= self.start_date)]
        base_table.reset_index(drop=True, inplace=True)

        # Create 30 and 7 day returns
        base_table['30_DAY_AVG_RETURNS'] = base_table['RETURN'].rolling(30, min_periods=30).mean()
        base_table['15_DAY_AVG_RETURNS'] = base_table['RETURN'].rolling(15, min_periods=15).mean()
        base_table['5_DAY_AVG_RETURNS'] = base_table['RETURN'].rolling(5, min_periods=5).mean()

        # Get MFI and MFI momentum
        base_table['MFI'] = self.MFI_calculator(base_table)
        base_table['MFI_INTEGRATED'] = self.integrate_series(base_table['MFI'])

        # Create stochastic oscillator
        base_table = base_table.merge(self.stochastic_oscillator(base_table, 15), how='inner', left_on='DATE', right_on='DATE')
        base_table['OSCILLATOR_DIVERGENCE'] = base_table['15_DAY_OSCILLATOR'] - base_table['OSCILLATOR_AVG']
        base_table['OSCILLATOR_INTEGRATED'] = self.integrate_series(base_table['15_DAY_OSCILLATOR'])

        # Create the coitnegration measurement to be used as a regressor
        base_table['COINT_SCORE'], cointegration_set = self.fit_cointegration_models(base_table)
        base_table['COINT_INTEGRATED'] = self.integrate_series(base_table['COINT_SCORE'])

        #add the SPY Futures price
        spy_futures = self.stock_dict['ES=F'][['DATE', 'RETURN']]
        spy_futures.rename(columns={'RETURN': 'SPY_FUTURES'}, inplace=True)
        base_table = base_table.merge(spy_futures, how='left', on='DATE')
        base_table.fillna(0, inplace=True)

        # Subset base table because first 30 rows have no substantive information
        base_table = base_table[29:].reset_index(drop=True)
        
        return base_table, cointegration_set

    @staticmethod
    def label_observations(base_table, upper_triggers, lower_triggers, mean_return, horizon=5):
        """
        Labeling observations, given triple barrier method. Tripple barrier method labels depending on
        whether upper (buy), lower (sell), or horizontal (hold) triggers are intersected first.

        :param base_table:
        :param upper_triggers:
        :param lower_triggers:
        :param mean_return:
        :param horizon:

        :return labels
        :return strike_dates:
        """
        labels = []
        strike_dates = []
        date_ranges = []

        for ind, row in base_table.iterrows():
            buy = row['OPEN']

            upper_barrier = (upper_triggers[ind])
            upper_barriers = [upper_barrier + mean_return * (x + 1) for x in range(horizon)]
            upper_bound = np.add(np.multiply(upper_barriers, buy), buy)

            lower_barrier = (lower_triggers[ind])
            lower_barriers = [lower_barrier + mean_return * (x + 1) for x in range(horizon)]
            lower_bound = np.add(buy, np.multiply(lower_barriers, buy))

            forward_obs = base_table[ind:ind + horizon].reset_index(drop=True)

            control_date = pd.to_datetime('today')
            try:
                upper_pass = forward_obs['DATE'][np.argmin(np.where(forward_obs['CLOSE'] > upper_bound))]
            except:
                upper_pass = control_date
            try:
                lower_pass = forward_obs['DATE'][np.argmin(np.where(forward_obs['CLOSE'] < lower_bound))]
            except:
                lower_pass = control_date

            if lower_pass < upper_pass:
                labels.append('SELL')
                strike_dates.append(lower_pass)
            elif upper_pass < lower_pass:
                labels.append('BUY')
                strike_dates.append(upper_pass)
            else:
                labels.append('HOLD')
                strike_dates.append(max(forward_obs['DATE']))

        return labels, strike_dates

    def fit_cointegration_models(self, base_data):
        """
        This function runs cointegration tests on the entire set of tickers in our dictionary and returns the
        residual sets of coitnegrated series OLS fit... basically an identifier for statistical arbitrage

        :param base_data: data frame, the base table for the stock

        :return cointegration_residuals: the residual set, displaying arbitrage opportunities
        """
        # set empty containers to hold results of cointegration tests
        r_square_container = []
        cointegration_set = []
        cointegration_fit_container = []

        log_start = pd.to_datetime('today')
        X = base_data[['DATE', 'CLOSE']]

        # initialize loop to test each price_data for cointegration, drop stock when finished to reduce runs
        # and redundancy of creating pairwise relationships several times. Stop when all stocks are dropped
        # loop through all potential relationships, creating coint tests
        coint_test_keys = list(self.stock_dict.keys())
        coint_test_keys.remove(self.stock)
        for every in coint_test_keys:
            try:
                y=self.stock_dict[str(every)][['DATE', 'CLOSE']]
                # We only use stocks that have at least the length of data of our focus subject
                merge=pd.merge(X, y, how='inner', left_on='DATE', right_on='DATE')
                if len(merge) != len(X):
                    print('Lost some data on merge, rejecting cointegration test')
                    continue
                one=merge['CLOSE_x']
                two=merge['CLOSE_y']
                ci=sm.coint(one, two, trend='ct', maxlag=0)
                t=ci[1]
                # if t-score is beyond confidence threshold, regress the two stocks
                if t <= .05:
                    # print(str(every)+" is cointegrated")
                    two = sm.add_constant(two)
                    mod = OLS(one, two).fit()
                    se = mod.ssr**(1/2)
                    r_square = mod.rsquared_adj
                    if r_square < 0.70:
                        continue
                    cointegration_set.append(every)
                    r_square_container.append(r_square)
                    fitted_values = mod.fittedvalues.to_list()
                    cointegration_fit_container.append(fitted_values)
                    print('{} and {} are indicated as cointegrated'.format(every, self.stock))
            except:
                print("cointegration test for "+str(every)+" failed")

        if cointegration_fit_container:
            weighted_r_squared = np.divide(r_square_container, sum(r_square_container))
            cointegration_fit = sum(np.multiply(np.array(cointegration_fit_container).T, weighted_r_squared).T)
            cointegration_residuals = (X['CLOSE'] - cointegration_fit)

        else:
            cointegration_residuals = [0]*len(base_data)

        log_end = pd.to_datetime('today')
        print('Cointegration function began running at ' + str(log_start) + ' and ended at ' + str(log_end))
        return cointegration_residuals, cointegration_set

    def MFI_calculator(self, base_table):
        """
        Calculates the money flow index for a timeseries. An indicator of whether an asset is over-bought
        or over-sold by how much money is flowing in or out of an asset.

        :param base_table: data frame, the base table for the stock

        :return MFI: money flow index for each day
        """
        # create empty lists to store necessary values
        positive_flow = [0]
        negative_flow = [0]
        MFI = [0]

        # calculate average prices and money flow
        avg_price = base_table['AVG_PRICE']
        avg_price.reset_index(drop=True, inplace=True)
        money_flow = base_table['VOLUME']*avg_price
        money_flow.reset_index(drop=True, inplace=True)
        # loop through money flows, placing in either positive or negative flow depending on price value vs. lagged value
        for i in range(1, len(avg_price)-1):
            if avg_price[i] < avg_price[i-1]:
                positive_flow.append(0)
                negative_flow.append(money_flow[i])
            elif avg_price[i] > avg_price[i-1]:
                positive_flow.append(money_flow[i])
                negative_flow.append(0)
            else:
                positive_flow.append(0)
                negative_flow.append(0)
        # Calculate fourteen day money flows, with initial value then dropping+replacing through iterations
        flow_frame = pd.DataFrame({'POSITIVE':positive_flow, 'NEGATIVE':negative_flow})
        frtn_day_pos_flow = flow_frame['POSITIVE'].rolling(14, min_periods=1).sum()
        frtn_day_neg_flow = flow_frame['NEGATIVE'].rolling(14, min_periods=1).sum()
        MFI.extend(100-100/(1+frtn_day_pos_flow/frtn_day_neg_flow))
        return MFI

    def integrate_series(self, series):
        """
        Integrate a series that is passed through

        :param series: the series to run integration on

        :return integrated_series: the integrated time series
        """
        integrated_series = [0]
        offset_1 = series[:-1]
        dif = np.subtract(series[1:].reset_index(drop=True), offset_1).tolist()
        integrated_series.extend(dif)
        
        return integrated_series

    def stochastic_oscillator(self, base_data, day_range):
        """

        :param base_data:
        :param day_range:

        :return prices:
        """
        oscillators = base_data[['DATE', 'HIGH', 'LOW', 'CLOSE']]
        highs = oscillators['HIGH']
        lows = oscillators['LOW']
        X_day_high = highs.rolling(day_range, min_periods=1).max()
        X_day_low = lows.rolling(day_range, min_periods=1).min()
        end_row = len(oscillators)

        oscillators['{}_DAY_HIGH'.format(day_range)] = X_day_high
        oscillators['{}_DAY_LOW'.format(day_range)] = X_day_low
        oscillators['{}_DAY_OSCILLATOR'.format(day_range)] = (oscillators['CLOSE']-oscillators['{}_DAY_LOW'.format(day_range)])/\
            (oscillators['{}_DAY_HIGH'.format(day_range)]-oscillators['{}_DAY_LOW'.format(day_range)])
        oscillators['OSCILLATOR_AVG'] = oscillators['{}_DAY_OSCILLATOR'.format(day_range)].rolling(3, min_periods=1).mean()
        oscillators.drop(['HIGH', 'LOW', 'CLOSE', '{}_DAY_HIGH'.format(day_range), '{}_DAY_LOW'.format(day_range)], axis=1, inplace=True)
        return oscillators

from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import matplotlib.pyplot as plt

import hmm_fb_algorithm as hmm


class VolatilityModels:

    def __init__(self, timeseries):
        """

        :param timeseries: series, containing the volatility of returns
        """
        self.timeseries = timeseries

    def create_garch_model(self, integrate=True):
        """
        :return fitted_values: the fitted values of the model
        :return garch_model: fitted model
        """
        if integrate:
            endog = np.subtract(self.timeseries[1:].reset_index(drop=True), self.timeseries[:-1].reset_index(drop=True))
        else:
            endog = self.timeseries
        fitted_garch_model = SARIMAX(endog=endog, exog=None, order=[2, 0, 1]).fit(disp=False)
        fitted_values = fitted_garch_model.fittedvalues

        if integrate:
            fitted_values = np.add(fitted_values, self.timeseries[1:]).tolist()
            fitted_values.insert(0, 0)
        return fitted_values, fitted_garch_model

    def create_hmm_model(self, n_states, integrate=False):
        """

        :param n_states: number of states to model HMM under
        :param integrate: whether or not to integrate the timeseries

        :return fitted_values: the fitted values of the model
        :return fitted_hmm: fitted model
        """
        if integrate:
            endog = np.subtract(self.timeseries[:-1].reset_index(drop=True), self.timeseries[1:].reset_index(drop=True))
        else:
            endog = self.timeseries
        ss_model = hmm.HiddenMarkovModel(endog, n_states=n_states, scale_value=100)
        fitted_hmm = ss_model.fit_hmm()

        fitted_values = fitted_hmm.fitted_values

        if integrate:
            fitted_values = np.add(fitted_values, self.timeseries[1:]).tolist()
            fitted_values.insert(0, 0)

        return fitted_values, fitted_hmm

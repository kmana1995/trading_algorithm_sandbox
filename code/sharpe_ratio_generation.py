import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.optimize as opt


class SharpePredict:

    def __init__(self, volatility_model, predicted_probabilities, bound_limits):
        self.volatility_model = volatility_model
        self.predicted_probabilities = predicted_probabilities
        self.bound_limits = bound_limits

    def generate_sharpe_projection(self):
        """
        Generate a projected sharpe ratio for an asset
        """
        mu_estimate, sigma_estimate = self.evaluate_pricing_distribution()

        sharpe_projection = mu_estimate/sigma_estimate

        return sharpe_projection.item(), sigma_estimate

    def evaluate_pricing_distribution(self):
        """
        Evaluate the parameters of our return distribution (mu and sigma) based on the predicted buy probability
        and the estimated variance
        """
        # extract the probability of a buy from to forecast
        probability_of_buy = self.predicted_probabilities['BUY']

        # Extract the conditional volatility of said buy
        sigma_estimate = np.sqrt(self.volatility_model.forecast(steps=1))

        # Calculate the upper point forecast
        point = sigma_estimate*self.bound_limits

        # Build the distribution using the probabilities to discern a mu parameter
        mu_estimate = self.estimate_mu_parameter(probability_of_buy, point, sigma_estimate)

        return mu_estimate, sigma_estimate

    @staticmethod
    def estimate_mu_parameter(true_probability, point, sigma, mu_estimate=0.0, learning_rate=0.1):
        """

        :param true_probability: float, probability based on classifier
        :param point: float, point forecast for the probability to be true
        :param sigma: float, estimate of standard deviation estimate
        :param mu_estimate: float, initial mu estimate
        :param learning_rate: float, rate to control gradient
        :return:
        """
        cdf_true = 1-true_probability

        for i in range(50):
            cdf = norm.cdf(point, mu_estimate, sigma)
            loss = cdf - cdf_true
            mu_estimate = mu_estimate+loss*learning_rate

        return mu_estimate


import pickle
from collections import namedtuple
import logging
import os

import master_data_creation as mdc
import conditional_volatility_model as cvm
import classification_models as cm
import sharpe_ratio_generation as srg


class MasterRunResults:

    def __init__(self, stock, fitted_classifier_ensemble, class_predictions, sharpe_projection, current_price,
                 cointegration_set, estimated_variance):
        self.stock = stock
        self.fitted_classifier_ensemble = fitted_classifier_ensemble
        self.class_predictions = class_predictions
        self.sharpe_projection = sharpe_projection
        self.current_price = current_price
        self.cointegration_set = cointegration_set
        self.estimated_variance = estimated_variance

# Create master run function to perform data engineering, classification fit, classification predict for a single stock
class MasterRun:
    def __init__(self, data):
        self.data = data

    def master_run(self, stock):
        try:
            logging.info('Running for {}'.format(stock))
            current_price = self.data.stock_dict[stock]['ADJ_CLOSE'][-1:].item()

            # Fit conditional volatility models to Returns
            mean_returns = self.data.stock_dict[stock]['RETURN'].mean()
            variance_of_returns = ((self.data.stock_dict[stock]['RETURN']-mean_returns)**2)
            vm = cvm.VolatilityModels(variance_of_returns)
            #volatility, fitted_model = vm.create_hmm_model()
            fitted_volatility, fitted_model = vm.create_hmm_model(n_states=3)

            # Create master table, a collection of labels and key exogenous variables
            engineer = mdc.CreateMasterTable(stock, self.data, fitted_volatility, mean_returns, bound_limits=1.5)
            master_table, cointegration_set = engineer.base_table_create()

            # Fit classification models
            modeler = cm.ClassifierEnsemble(stock, master_table)
            fitted_ensemble = modeler.fit_classifier_ensemble()

            # Predict Classification
            current_state = master_table[-1:]
            predicted_probabilities = fitted_ensemble.predict(current_state)

            # Simulate returns based on the assigned classification and volatility estimates
            simulation = srg.SharpePredict(fitted_model, predicted_probabilities, bound_limits=1.5)
            sharpe_ratio_projection, sigma_estimate = simulation.generate_sharpe_projection()

            # Save the results
            results = MasterRunResults(stock, fitted_ensemble, predicted_probabilities, sharpe_ratio_projection,
                                       current_price, cointegration_set, sigma_estimate)
            self.save_master_results(stock, results)

        except Exception as e:
            logging.info('Master run failed on {}, failure was {}'.format(stock, e))

    def save_master_results(self, stock, results):
        file_path = open(os.path.join(self.data.results_data_path, '{}_results.p'.format(stock)), 'wb')
        pickle.dump(results, file_path)
        file_path.close()

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

import sampling_functions as sf


class ClassifierEnsembleResults:

    def __init__(self, fitted_models, min_max_scaler, model_accuracies, performance_df, stock):
        self.fitted_models = fitted_models
        self.min_max_scaler = min_max_scaler
        self.performance_df = performance_df
        self.model_accuracies = model_accuracies
        self.stock = stock

    def predict(self, data):
        """

        :param data: the data used to predict
        :return probabilities: the probabilities of each respective class
        """
        non_exog = ['DATE', 'LABEL', 'HIGH', 'STRIKE_DATE', 'LOW', 'OPEN', 'CLOSE', 'ADJ CLOSE', 'AVG_PRICE', \
                    'RETURN', 'STOCK']
        exog = data[data.columns[~data.columns.isin(non_exog)]][-1:]
        exog = self.min_max_scaler.transform(exog)

        pca_model = self.fitted_models['PCA']
        pca_exog = pca_model.transform(exog)

        models = list(self.fitted_models.keys())
        models.remove('PCA')

        p_buy = 0
        p_hold = 0
        p_sell = 0

        for model in models:
            fitted_model = self.fitted_models[model]
            classes = fitted_model.classes_
            if model in ['SVM', 'KNN']:
                prediction_probs = fitted_model.predict_proba(pca_exog)
            else:
                prediction_probs = fitted_model.predict_proba(exog)
            model_weight = self.model_accuracies[model]/self.model_accuracies['AGGREGATED_PERFORMANCE']
            prediction_probs = prediction_probs*model_weight
            p_buy += prediction_probs[:, np.where(classes == 'BUY')].item()
            p_hold += prediction_probs[:, np.where(classes == 'HOLD')].item()
            p_sell += prediction_probs[:, np.where(classes == 'SELL')].item()
        p_total = p_buy+p_hold+p_sell

        probabilities = pd.DataFrame({'STOCK': self.stock, 'BUY': p_buy/p_total, 'HOLD': p_hold/p_total, 'SELL': p_sell/p_total}, index=[0])

        return probabilities


class ClassifierEnsemble:

    def __init__(self, stock, master_table):
        self.stock = stock
        self.master_table = master_table

    def fit_classifier_ensemble(self, type='k_folds'):
        """
        Fitting a classification ensemble of a dataframe

        :param type: str, which type of validation to run ensemble validation on ('loocv' or 'k_folds')
        :return ClassifierEnsembleResults: the results, including predict ability
        """
        if type == 'loocv':
            model_accuracies, performance_df, fitted_models = self.run_loocv_fit()
        if type == 'k_folds':
            model_accuracies, performance_df, fitted_models, scaler = self.run_k_folds_fit()

        return ClassifierEnsembleResults(fitted_models, scaler, model_accuracies, performance_df, self.stock)

    def create_training_sets(self, population, non_exog):
        """
        This function creates a training set for the classifiers

        :param population: dataframe, the training population
        :param non_exog: list, variables that are not used to produce a fit
        :return train_x: array, exogenous training variables
        :return train_y: series, dependant training lables
        :return scaler: class, the min_max scaler used to normalize the data
        """

        # We offset the exogenous variables by one time step, to make them actionable
        exog = population.columns[~population.columns.isin(non_exog)]
        population[exog][1:] = population[exog][:-1]
        population = population[1:].reset_index(drop=True)

        # take samples from population
        sample = sf.run_sampling(self.stock, population, 'sequential', 1000)
        sample.reset_index(drop=True, inplace=True)

        # set exogenous and dependant variables
        exog = sample.columns[~sample.columns.isin(non_exog)]
        exog = sample[exog]
        scaler = MinMaxScaler(feature_range=(0, 1))
        exog = scaler.fit_transform(exog)
        train_x = exog
        train_y = sample['LABEL'].reset_index(drop=True)

        return train_x, train_y, scaler

    def run_k_folds_fit(self, fold_size=25):
        """
        Performing k_folds validation to determine model performance

        :param fold_size: int, size of validation fold
        :return model_performance: dictionary, individual model performance
        :return performance_df: dataframe, the individual model performance
        :return fitted_models: dictionary, containing the fitted classifiers
        :return scaler: class, the min_max scaler used to normalize the data
        """
        # create model fit fold
        fit_population = self.master_table[:-fold_size]
        non_exog = ['DATE', 'LABEL', 'HIGH', 'STRIKE_DATE', 'LOW', 'OPEN', 'CLOSE', 'ADJ CLOSE', 'AVG_PRICE', \
                    'RETURN', 'STOCK']
        train_x, train_y, scaler = self.create_training_sets(fit_population, non_exog)

        # fit models
        fitted_models = self.fit_classifiers(train_x, train_y)

        # create model test fold
        test = self.master_table[-(fold_size + 1):-1]

        exog = test.columns[~test.columns.isin(non_exog)]
        exog = test[exog]
        exog = scaler.transform(exog)

        # predict
        labels = self.master_table['LABEL'][-fold_size:].tolist()
        model_performance, performance_df = self.predict_classification(exog, fitted_models, labels)

        # Perform final fit on models
        train_x, train_y, scaler = self.create_training_sets(self.master_table, non_exog)
        fitted_models = self.fit_classifiers(train_x, train_y)

        return model_performance, performance_df, fitted_models, scaler

    def run_loocv_fit(self, n_runs=50):
        """
        Run LOOCV to determine learner performance. This model is particularly time consuming and k_folds is preferred
        in most cases

        :param n_runs:
        :return model_performance: dictionary, individual model performance
        :return performance_df: dataframe, the individual model performance
        :return fitted_models: dictionary, containing the fitted classifiers
        :return scaler: class, the min_max scaler used to normalize the data
        """
        predictions = []

        indexers = list(range(n_runs))
        indexers.reverse()
        indexers = np.add(indexers, 1)

        for i in indexers:
            fit_population = self.master_table[:-(i + 1)]
            non_exog = ['DATE', 'LABEL', 'HIGH', 'STRIKE_DATE', 'LOW', 'OPEN', 'CLOSE', 'ADJ CLOSE', 'AVG_PRICE',\
                        'RETURN', 'STOCK']
            train_x, train_y, scaler = self.create_training_sets(fit_population, non_exog)

            # fit models
            fitted_models = self.fit_classifiers(train_x, train_y)

            # forward step one for LOO sample
            test = self.master_table[-(i+1):-i]

            exog = test.columns[~test.columns.isin(non_exog)]
            exog = test[exog]
            exog = scaler.transform(exog)

            # predict
            iter_predictions = self.predict_classification(exog, fitted_models)
            predictions.append(iter_predictions)

        performance_df = pd.concat(predictions)
        labels = self.master_table['LABEL'][-n_runs:].tolist()
        performance_df['ACTUAL_CLASS'] = labels
        models = performance_df.columns.tolist()
        models.remove('ACTUAL_CLASS')

        model_performance_dict = {}
        aggregate_performance = 0
        for model in models:
            model_performance = np.sum(np.where(performance_df[model] == \
                                                        performance_df['ACTUAL_CLASS'], 1, 0)) / len(performance_df)
            model_performance_dict.update({'{}_PERFORMANCE'.format(model): model_performance})
            aggregate_performance += model_performance
        model_performance_dict.update({'AGGREGATE_PERFORMANCE': aggregate_performance})
        return model_performance, performance_df, fitted_models

    @staticmethod
    def predict_classification(exog, fitted_models, labels=None):
        """
        Predicting classes for each respective validation model (k-folds and LOOCV) This function IS NOT
        to be confused with the final predict method for our classifier ensemble

        :param exog: array of exogenous variable
        :param fitted_models: dictionary, containing the fitted models
        :param labels: the labels (classifications) of our test set

        :return predictions:
        """
        pca_model = fitted_models['PCA']
        pca_exog = pca_model.transform(exog)

        models = list(fitted_models.keys())
        models.remove('PCA')
        predictions = pd.DataFrame()

        for model in models:
            fitted_model = fitted_models[model]
            if model in ['SVM', 'KNN']:
                model_predict = fitted_model.predict(pca_exog).tolist()
            else:
                model_predict = fitted_model.predict(exog).tolist()
            predictions['{}_PREDICTION'.format(model)] = model_predict

        if labels is not None:
            predictions['ACTUAL_CLASS'] = labels

            model_performance_dict = {}
            performance_df = pd.DataFrame()
            aggregate_performance = 0
            for model in models:
                model_performance = np.sum(np.where(predictions['{}_PREDICTION'.format(model)] == \
                                                        predictions['ACTUAL_CLASS'], 1, 0)) / len(predictions)
                model_performance_dict.update({model: model_performance})
                aggregate_performance += model_performance
                performance_df['{}_PERFORMANCE'.format(model)] = [np.sum(np.where(predictions['{}_PREDICTION'.format(model)] == \
                                                        predictions['ACTUAL_CLASS'], 1, 0)) / len(predictions)]
            model_performance_dict.update({'AGGREGATED_PERFORMANCE': aggregate_performance})
            return model_performance_dict, performance_df
        else:
            return predictions


    def fit_classifiers(self, train_x, train_y):
        """

        :param train_x: array, training set
        :param train_y: series, training labels

        :return fitted_models: the fitted classifiers
        """
        fitted_models = {}

        # Fit PCA to reduce feature space
        pca_model = self.run_pca(train_x)
        pca_train_x = pca_model.transform(train_x)
        fitted_models.update({'PCA': pca_model})

        # Fit KNN and SVM on principle components
        # support_vector_machine = self.create_svm_model(pca_train_x, train_y)
        # fitted_models.update({'SVM': support_vector_machine})

        knn_model = self.create_bagged_knn(pca_train_x, train_y)
        fitted_models.update({'KNN': knn_model})

        # Fit random forest and perceptron on raw data, handling large feature spaces well
        random_forest = self.create_random_forest(train_x, train_y)
        fitted_models.update({'RANDOM_FOREST': random_forest})

        perceptron_model = self.create_perceptron(train_x, train_y)
        fitted_models.update({'PERCEPTRON': perceptron_model})

        return fitted_models

    @staticmethod
    def create_svm_model(train_x, train_y):
        """

        :param train_x:
        :param train_y:

        :return support_vector_machine:
        """
        support_vector_machine = SVC(C=1, kernel='rbf', probability=True, random_state=1)
        support_vector_machine.fit(X=train_x, y=train_y)
        return support_vector_machine

    @staticmethod
    def create_random_forest(train_x, train_y):
        """

        :param train_x:
        :param train_y:

        :return tree:
        """
        tree = RandomForestClassifier(n_estimators=25, criterion='entropy', random_state=1, oob_score=True)
        tree.fit(train_x, train_y)
        return tree

    @staticmethod
    def create_bagged_knn(train_x, train_y):
        """

        :param train_x:
        :param train_y:

        :return knn:
        """
        knn = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=25)
        knn.fit(X=train_x, y=train_y)
        return knn

    @staticmethod
    def create_perceptron(train_x, train_y):
        """

        :param train_x:
        :param train_y:

        :return knn:
        """
        perceptron = MLPClassifier(activation='logistic', solver='sgd')
        perceptron.fit(X=train_x, y=train_y)
        return perceptron

    @staticmethod
    def run_pca(train_x):
        """
        Run principle components analysis to shrink our feature space, necessary for performance of certain models
        (KNN, SVM (not used))

        :param train_x:
        :return:
        """
        pca = PCA(n_components=5)
        pca = pca.fit(train_x)
        return pca

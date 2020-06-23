from pyomo.environ import *
import glob
import pickle
import pandas as pd
import os

class LinearOptimization:

    def __init__(self, results_dictionary):
        self.results_dictionary = results_dictionary

    def create_data(self):

        # create a set of stocks
        stock_set = list(self.results_dictionary.keys())

        # create a set of cointegration pairs
        cointegration_set = []
        for stock in stock_set:
            cointegration_pairs = self.results_dictionary[stock].cointegration_set
            for member in cointegration_pairs:
                if member not in stock_set:
                    continue
                cointegration_set.append((stock, member))

        # create sharpe ration parameters
        sharpe_ratio_param = {}
        for stock in stock_set:
            sharpe_ratio = {stock: self.results_dictionary[stock].sharpe_projection}
            sharpe_ratio_param.update(sharpe_ratio)

        # create the parameter of the current set
        current_price_param = {}
        for stock in stock_set:
            current_price = {stock: self.results_dictionary[stock].current_price}
            current_price_param.update(current_price)

        data = {None: {
            'stock_set': {None: stock_set},
            'cointegration_set': {None: cointegration_set},
            'sharpe_ratio_param': sharpe_ratio_param,
            'current_price_param': current_price_param
        }}

        return data

    @staticmethod
    def objective_statement(model):
        """
        define the objective statement of the optimization model
        """
        return sum(model.purchase_quantity[stock] * model.current_price_param[stock] * model.sharpe_ratio_param[stock]
                   for stock in model.stock_set)

    @staticmethod
    def constraint_max_single_asset(model, stock):
        """
        Establish a min diversity threshold
        """
        return model.purchase_quantity[stock] * model.current_price_param[stock] <= 1000

    @staticmethod
    def constraint_total_purchase_power(model):
        """
        We only have a certain amount of capital for investing
        """
        return sum(model.purchase_quantity[stock] * model.current_price_param[stock] for stock in model.stock_set) <= 5000

    def create_abstract_model(self):

        model = AbstractModel()

        model.stock_set = Set(dimen=1)
        model.cointegration_set = Set(dimen=2)
        model.sharpe_ratio_param = Param(model.stock_set, within=Reals)
        model.current_price_param = Param(model.stock_set, within=PositiveReals)
        model.purchase_quantity = Var(model.stock_set, within=NonNegativeIntegers)

        # set objective statement
        model.objective = Objective(rule=self.objective_statement, sense=maximize)

        # set constraints
        model.constraint_max_single_asset = Constraint(model.stock_set, rule=self.constraint_max_single_asset)
        model.constraint_purchase_power = Constraint(rule=self.constraint_total_purchase_power)

        return model


    def run_optimization(self):
        data = self.create_data()
        model = self.create_abstract_model()
        model_instance = model.create_instance(data)

        optimize = SolverFactory('glpk')
        solution = optimize.solve(model_instance)

        solution_df = []
        print(str(solution.solver.termination_condition))
        if str(solution.solver.termination_condition) in 'optimal':
            print('Optimal sharpe ratio estimated at {}'.format(model_instance.objective.expr()))
            for stock in model_instance.stock_set:
                print("Purchase {} of stock {} at {}".format(model_instance.purchase_quantity.get_values()[stock],
                                                             stock, model_instance.current_price_param[stock]))
                purchase = pd.DataFrame({'STOCK': stock, 'N_SHARES': model_instance.purchase_quantity.get_values()[stock],
                                         'PRICE': model_instance.current_price_param[stock],
                                         'EST_VARIANCE': self.results_dictionary[stock].estimated_variance}, index=[0])
                purchase['UPPER_BOUND'] = purchase['PRICE']*1+purchase['EST_VARIANCE']
                purchase['LOWER_BOUND'] = purchase['PRICE']*1-purchase['EST_VARIANCE']
                solution_df.append(purchase)
        solution_df = pd.concat(solution_df)
        solution_df = solution_df.sort_values('N_SHARES', ascending=False)

        return solution_df

if __name__ == '__main__':
    # break out of parallelization and unpack results
    result_files = glob.glob(r'C:\Users\kylea\OneDrive\Documents\Stock_DB\Master_Run_Results\*')
    results_dictionary = {}
    for file in result_files:
        path = open(file, 'rb')
        results = pickle.load(path)
        stock = results.stock
        results_dictionary.update({stock: results})
        path.close()

    # optimize
    optimization = LinearOptimization(results_dictionary)
    optimization.run_optimization()
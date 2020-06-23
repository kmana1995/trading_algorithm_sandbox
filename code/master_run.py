import pandas as pd
import multiprocessing as mp
from functools import partial
import logging
import os
import glob
import pickle

import master_run_functions as mrf
import base_data_layer as bd
import logger
import optimization_model

def mp_handler(data, n_cores=4):
    """
    Multi-processing handler for our master run, for time efficiency.


    :param data: initialized data
    :param n_cores: int, number of cores to compute across
    :return:
    """
    logger.log_setup()

    # Multi-process the master run function
    master = mrf.MasterRun(data)
    p = mp.Pool(n_cores)
    func = partial(master.master_run)
    p.map(func, data.tickers)


    # break out of parallelization and unpack results
    result_files = glob.glob(os.path.join(data.results_data_path, '*'))
    results_dictionary = {}
    for file in result_files:
        path = open(file, 'rb')
        results = pickle.load(path)
        stock = results.stock
        results_dictionary.update({stock: results})
        path.close()

    # optimize
    optimization = optimization_model.LinearOptimization(results_dictionary, available_to_invest=5000)
    optimal_buys = optimization.run_optimization()
    print('Optimal buys and bounds on five day horizon for SP500 are:')
    print(optimal_buys)


if __name__ == '__main__':
    # initialize the logger
    logger.log_setup()
    logger.truncate_logger()
    logging.info('Logger Start')

    run_type = 'TEST' # TEST or FULL_RUN
    start_date = (pd.to_datetime('today') - pd.DateOffset(days=730))
    end_date = pd.to_datetime('today')
    data = bd.DataLayer(run_type, start_date, end_date)
    data.tickers = data.tickers[:8]
    master = mrf.MasterRun(data)
    #master.master_run('MSFT')
    mp_handler(data, n_cores=8)
import pandas_datareader as pdr
import pandas as pd
import pickle
import logging
import os
import tempfile as tmp
import csv


class DataLayer:
    def __init__(self, run_type, start, end):
        """

        :param run_type: str, which run time is being performed (TEST, BENCHMARK, FULL_RUN)
        :param start: datetime, the start date we are pulling the timeseries for
        :param end: datetime, the end date we are pulling the timeseries for
        """
        self.results_data_path = tmp.mkdtemp(prefix='master_results')
        self.tickers = self.load_sp500_tickers()
        self.stock_dict = self.pull_stock_data(run_type, start, end)
        self.start_date = start
        self.end_date = end


    def load_sp500_tickers(self):
        '''
        Loads tickers for the S&P 500 from csv file
        :return: sp_500: list, tickers of the S&P 500
        '''

        stocks = pd.read_csv(os.path.join(os.path.dirname(__file__), 'SP500.csv'))
        stocks.columns = stocks.columns.str.upper().str.replace(' ', '_')
        tickers = stocks['SYMBOL']

        return tickers

    def pull_stock_data(self, run_type, start, end):
        '''
        Function to pull the price data for defined tickers and store them within a dictionary
        while writing them to csv in the defined file path

        :param run_type: str, which run time is being performed (TEST, BENCHMARK, FULL_RUN)
        :param start: datetime, the start date we are pulling the timeseries for
        :param end: datetime, the end date we are pulling the timeseries for
        :return stock_dict: dictionary containing all of the stock prices
        '''

        logging.info('Starting base data pull at {}'.format(pd.to_datetime('now')))

        # create empty dictionary and list to be filled with data frames containing time series data
        stock_dict = {}
        master_price_data = []

        # iterate through the tickers to populate data frame with time series actuals from the last year
        for stock in self.tickers:
            try:
                f=pdr.DataReader(stock, 'yahoo', start, end)
                f.reset_index(inplace=True)
                f.columns = f.columns.str.upper()
                f.columns = f.columns.str.replace(' ', '_')
                f['DATE'] = pd.to_datetime(f['DATE'])
                f = f[f['OPEN'] > 0].reset_index(drop=True)
                f['AVG_PRICE'] = (f['HIGH']+f['LOW']+f['CLOSE'])/3
                f['RETURN'] = (f['CLOSE'] - f['OPEN'])/f['OPEN']
                f['STOCK'] = stock
                stock_dict.update({str(stock): f})
                master_price_data.append(f)
            except:
                logging.info('{} failed to pull price data from API'.format(stock))

        logging.info('Ending base data pull at {}'.format(pd.to_datetime('now')))
        return stock_dict
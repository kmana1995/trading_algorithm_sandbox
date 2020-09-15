# trading_algorithm_sandbox
This code repository is my sandbox for playing around with financial ML and optimization methods. This algorithm is by no means ready for millions from an investor group... *sarcasm*

Running the master_run.py file will run the entire program, consisting of:

  1. Pulling price data from yahoo finance API for the sp500<br>
  3. Running conditional volatility model (hidden markov or GARCH model) on the variance of returns<br>
  2. Pre-processing data to create features that can be used as exogenous variables in classification, intended to differentiate buy, sell,      and hold opportunities. Features include:<br>
      a. MFI<br>
      b. Stochastic Oscillator/Oscillator Divergence<br>
      c. A metric called "Cointegration Score", which is the residual set of every cointegrated pair between the asset and the sp500<br>
      d. Average Returns (30, 15, 5 day)<br><br>
  3. Fitting a classifier ensemble on the data<br>
  4. Predicting the class and the current volatility of an asset<br>
  5. Setting the upper, lower, and horizontal bounds of a potential buy<br>
  6. Running an LP to determine the optimal buy portfolio<br>
  

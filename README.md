Must
* Forecasting Statistics
    * Exponential Smoothing with Darts
    * ARIMA with Darts

Should
* Python for Finance Chapter 7

Could
* sktime AutoETS and AutoArima



---
### Best Practices
#### Preparation
* Convert to log returns

#### Exploration 
* Outlier detection
* Change points
* Trending, mean reverting, or random?   Hurst
* Stylized Facts
    * Non-normal returns?  PDF and QQ plots
    * Volatility clustering?
    * Absence of autocorrelation
    * Small decreasing autocorrelation in squared returns
    * Leverage effect

---
### Tools
* darts
    * ExponentialSmoothing uses statsmodels, which does not support seconds.

sktime.forecasting

---
### Forecasting Approaches
#### Statistical  
* Exponential Smoothing
* ARIMA

#### Probabilitistic    
* Markov Models
* Bayesian Inference

#### Machine Learning    
* Regression
* Vector Regression
* Decision Trees/Random Forests

#### Deep Learning    
* LSTM
* GRU

#### Reinforcement Learning    
* SB3 PPO
    * Plateau'd at 50k timesteps
    * Should make episodes variable
* SAC




import numpy as np
import pandas as pd
import scipy.stats as scs
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
import warnings

# settings 
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
warnings.simplefilter(action='ignore', category=InterpolationWarning)

def adf_test(x):
    '''
    Function for performing the Augmented Dickey-Fuller test for stationarity
    
    Null Hypothesis: time series is not stationary
    Alternate Hypothesis: time series is stationary

    Parameters
    ----------
    x : pd.Series / np.array
        The time series to be checked for stationarity
    
    Returns
    -------
    results: pd.DataFrame
        A DataFrame with the ADF test's results
    '''

    indices = ['Test Statistic', 'p-value',
               '# of Lags Used', '# of Observations Used']

    adf_test = adfuller(x, autolag='AIC')
    results = pd.Series(adf_test[0:4], index=indices)

    for key, value in adf_test[4].items():
        results[f'Critical Value ({key})'] = value

    return results

def kpss_test(x, h0_type='c'):
    '''
    Function for performing the Kwiatkowski-Phillips-Schmidt-Shin test for stationarity

    Null Hypothesis: time series is stationary
    Alternate Hypothesis: time series is not stationary

    Parameters
    ----------
    x: pd.Series / np.array
        The time series to be checked for stationarity
    h0_type: str{'c', 'ct'}
        Indicates the null hypothesis of the KPSS test:
            * 'c': The data is stationary around a constant(default)
            * 'ct': The data is stationary around a trend
    
    Returns
    -------
    results: pd.DataFrame
        A DataFrame with the KPSS test's results
    '''

    indices = ['Test Statistic', 'p-value', '# of Lags']

    kpss_test = kpss(x, regression=h0_type)
    results = pd.Series(kpss_test[0:3], index=indices)

    for key, value in kpss_test[3].items():
        results[f'Critical Value ({key})'] = value

    return results

def test_autocorrelation(x, n_lags=40, alpha=0.05, h0_type='c'):
    '''
    Function for testing the stationarity of a series by using:
    * the ADF test
    * the KPSS test
    * ACF/PACF plots

    Parameters
    ----------
    x: pd.Series / np.array
        The time series to be checked for stationarity
    n_lags : int
        The number of lags for the ACF/PACF plots
    alpha : float
        Significance level for the ACF/PACF plots
    h0_type: str{'c', 'ct'}
        Indicates the null hypothesis of the KPSS test:
            * 'c': The data is stationary around a constant(default)
            * 'ct': The data is stationary around a trend

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the ACF/PACF plot
    '''

    adf_results = adf_test(x)
    kpss_results = kpss_test(x, h0_type=h0_type)

    print('ADF test statistic: {:.2f} (p-val: {:.2f})'.format(adf_results['Test Statistic'],
                                                             adf_results['p-value']))
    print('KPSS test statistic: {:.2f} (p-val: {:.2f})'.format(kpss_results['Test Statistic'],
                                                              kpss_results['p-value']))

    fig, ax = plt.subplots(2, figsize=(16, 10))
    plot_acf(x, ax=ax[0], lags=n_lags, alpha=alpha)
    plot_pacf(x, ax=ax[1], lags=n_lags, alpha=alpha)
    return fig

def create_sinewave():
    date = pd.to_datetime("2024-01-02")
    start = date + pd.to_timedelta("09:30:00")
    end = date + pd.to_timedelta("15:59:59")

    num_seconds = int((end - start).total_seconds()) + 1
    period = 390 * 60  # Period of the sine wave, in seconds (e.g., one day)
    amplitude = 5  # Amplitude of the sine wave
    base_price = 472.5  # Base price around which the sine wave oscillates

    # Generate sine wave for the close prices
    t = np.arange(num_seconds)
    close_prices = base_price + amplitude * np.sin(2 * np.pi * t / period)

    # Derive open, high, and low prices from close prices
    open_prices = close_prices + np.random.uniform(-0.1, 0.1, size=num_seconds)
    high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(0.0, 0.2, size=num_seconds)
    low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(0.0, 0.2, size=num_seconds)

    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
    }, index=pd.date_range(start=start, end=end, freq='1s'))
    df.index.name = 'datetime'

    df["log_rtn"] = np.log(df["close"]/df["close"].shift(1))
    df = df[["close", "log_rtn"]].dropna()

    return df

def get_hurst_exponent(ts, max_lag=20):
    """Returns the Hurst Exponent of the time series"""
    
    lags = range(2, max_lag)

    # standard deviations of the lagged differences
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]

    # calculate the slope of the log plot -> the Hurst Exponent
    hurst_exp = np.polyfit(np.log(lags), np.log(tau), 1)[0]

    return hurst_exp


# def print_descriptive_stats(df):
#     print("---------- Descriptive Statistics ----------")
#     print("Range of dates:", min(df.index.date), "-", max(df.index.date))
#     print("Number of observations:", df.shape[0])
#     print(f"Mean: {df.log_rtn.mean():.4f}")
#     print(f"Median: {df.log_rtn.median():.4f}")
#     print(f"Min: {df.log_rtn.min():.4f}")
#     print(f"Max: {df.log_rtn.max():.4f}")
#     print(f"Standard Deviation: {df.log_rtn.std():.4f}")
#     print(f"Skewness: {df.log_rtn.skew():.4f}")
#     print(f"Kurtosis: {df.log_rtn.kurtosis():.4f}") 
#     jb_test = scs.jarque_bera(df["log_rtn"].values)
#     print(f"Jarque-Bera statistic: {jb_test[0]:.2f} with p-value: {jb_test[1]:.2f}")
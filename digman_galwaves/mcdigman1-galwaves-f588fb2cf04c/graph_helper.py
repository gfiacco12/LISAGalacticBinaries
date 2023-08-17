import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics import tsaplots

def makeHistogramofLogLike(data):
    plt.hist(data)
    plt.title("Loglikelihood of Samples")
    plt.show()
    return

def makeScatterPlot(data_x, data_y):
    plt.figure()
    plt.scatter(data_x, data_y)
    plt.title("Scatter Plot of LogL vs $\dot{f}$")
    plt.xlabel("Log likelihood")
    plt.ylabel("$\dot{f}$")
    plt.show()
    return

def plotAutoCorrelationLength(data, lags):
    plt.figure()
    tsaplots.plot_acf(data, lags=lags)
    plt.title("Autocorrelation of data")
    plt.xlabel("h")
    plt.ylabel("Correlation Coefficient")
    plt.show()
    return
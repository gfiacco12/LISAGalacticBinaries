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
    plt.title("Scatter Plot of LogL vs alpha - Mc, Iwd Tides Model")
    plt.xlabel("Log likelihood")
    plt.ylabel("alpha")
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

def plotChains(data, N_blocks):
    plt.figure()
    for i in range(len(data[1])):
        chains = np.array(data)[:,i]
        iter = np.linspace(0, len(data[0])*1000, N_blocks)
        plt.plot(iter, chains, label="Chain %s" % i)
    plt.title("Trace Plot")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("LogL")
    plt.show()
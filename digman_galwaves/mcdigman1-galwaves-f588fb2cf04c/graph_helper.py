import numpy as np
import math
from scipy import optimize
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics import tsaplots
from wdm_const import *

def makeHistogramofLogLike(data):
    plt.hist(data)
    plt.title("Loglikelihood of Samples")
    plt.savefig("LogL values")
    return

def makeScatterPlot(data_x, data_y):
    plt.figure()
    plt.scatter(data_x, data_y)
    plt.title("Scatter Plot of LogL vs alpha - Mc, Iwd Tides Model")
    plt.xlabel("Log likelihood")
    plt.ylabel("alpha")
    plt.show()
    return

def plotAutoCorrelationLength(data, lags, param_name):
    plt.figure()
    tsaplots.plot_acf(data, lags=lags)
    plt.title("Autocorrelation of %s" % param_name)
    plt.xlabel("h")
    plt.ylabel("Correlation Coefficient")
    plt.show()
    return

def plotChains(data):
    plt.figure()
    N_blocks = len(data)
    N_chains = len(data[0])
    for i in range(N_chains - 1):
        chains = np.array(data)[:,i]
        iter = np.linspace(0, (N_blocks - 1)*1000, N_blocks)
        plt.plot(iter, chains)
    plt.title("Trace Plot")
    plt.legend()
    plt.ylim(-1e5, 7e5)
    plt.xlabel("Iterations")
    plt.ylabel("LogL")
    plt.savefig("chains fig")

def betadelta_m1m2_check(beta, delta, freq0, tobs, mc, mt):
    params_true = np.array([mc / MSOLAR, mt / MSOLAR])      
    #starting guess for masses
    fdot = beta/(tobs**2)
    chirp_guess = (fdot * 5 / (96 * (np.pi**(8/3)) * (freq0**(11/3))))**(3./5)
    total_guess = chirp_guess / (0.24)**(3./5)
    final_guess = getRootFinder_tides_chirpTotalMass(freq0, beta, delta, tobs, params_true[0], params_true[1], chirp_guess, total_guess)
    #filter the masses
    if final_guess.success == True:
        if np.isnan(final_guess.x[0]) == False and np.isnan(final_guess.x[1]) == False:
            eta = (final_guess.x[0]/final_guess.x[1])**(5./3.)
            if eta <= 0.25:
                m1, m2 = get_comp_mass_Mc_Mt(final_guess.x[0], final_guess.x[1])
                if m2 < m1:
                    if np.isnan(m1) == False and np.isnan(m2) == False:
                        return True
    return False

def get_comp_mass_Mc_Mt(Mc, Mt):
    eta = (Mc/Mt)**(5./3.)
    qq = ((1-2.*eta) - np.sqrt((1-2.*eta)**2. - 4.*eta**2.))/(2.*eta)
    M1=Mt/(1.+qq)
    M2=qq*Mt/(1.+qq)
    q = M2 / M1
    return M1, M2

def getFrequency_McMt_Tides(freq0, t_obs, params, results_exact):
    chirpMass, totalMass = params
    eta = ( (chirpMass/MSOLAR) / (totalMass/MSOLAR) )**(5/3)
    x = (np.pi*freq0*totalMass)**(2/3)

    dm = (1-(4*eta))**(1/2)

    if math.isnan(dm):
        dm = 1.e-30
    
    mass1 = totalMass * (1 + dm) / 2
    mass2 = totalMass * (1 - dm) / 2

    #moments of inertia
    I_wd = 8.51e-10 * ((mass1/(0.6*MSOLAR))**(1/3) + (mass2/(0.6*MSOLAR))**(1/3))
    I_orb = chirpMass**(5/3) / ((np.pi*freq0)**(4/3))

    #tidal deformability parameters
    r1 = 10**9 * (mass1/(0.6 * MSOLAR))**(-1/3) * (1/CLIGHT)
    r2 = 10**9 * (mass2/(0.6 * MSOLAR))**(-1/3) * (1/CLIGHT)
    y1 = (2/3) * 0.1 * (r1**5)
    y2 = (2/3) * 0.1 * r2**5
    lam1 = y1 / (mass1**5)
    lam2 = y2 / (mass2**5)
    lam = (8/13) * ((1 + 7*eta - 31*(eta**2))*(lam1 + lam2) + np.sqrt(1 - 4*eta)*(1 + 9*eta - 11*(eta**2))*(lam1 - lam2))

    #corrections to fdot
    deltaI = ((3*I_wd/I_orb)/(1 - (3*I_wd/I_orb)))
    deltaLam = (39 / 8) * lam
    dlx = deltaLam*(x**5)
    #corrections to fddot
    ddeltaI = (26/11)*deltaI + (19/11)*deltaI**2
    ddeltaLam = (32/11) * deltaLam * x**5 + (21/11) * deltaLam**2 * x**10
    
    fdot_pp = 96/5*np.pi**(8/3)*freq0**(11/3)*chirpMass**(5/3)

    #Tides + deformability freqD and freqDD + deltaLam*(x**5)
    fdot = fdot_pp * (1 + deltaI )
    fddot = (11/3)*(fdot_pp**2/freq0) * (1 + ddeltaI )
    #parameterize frequencies
    alpha = freq0*t_obs
    beta = fdot*(t_obs**2)
    gamma = fddot*(t_obs**3)
    delta = (gamma - (11/3)*((beta**2) / (alpha)))

    F=np.zeros(2)
    F[0] = (beta/ results_exact[0]) - 1
    F[1] = delta - results_exact[1]
    return F

def getRootFinder_tides_chirpTotalMass(freq0, beta, delta, t_obs, chirp_exact, total_exact, chirp_guess, total_guess):
    #root finding method for Mt and Mc from the tide GW equations
    params_guess = [chirp_guess, total_guess]

    params_exact = [chirp_exact, total_exact]

    results_exact = [beta, delta]

    #do the iteration
    fx = lambda p : getFrequency_McMt_Tides(freq0, t_obs, p, results_exact)

    final_guess = optimize.root(fx, params_guess, tol=1.e-10)
    
    for i in range(len(final_guess.x)):
        final_guess.x[i] /= MSOLAR
        params_exact[i] /= MSOLAR

    return final_guess
"""C 2023 Matthew C. Digman
helpers for computing the temperature ladder for parallel tempering"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import cumtrapz


class TemperatureLadder():
    """store a temperature ladder for parallel tempering"""

    def __init__(self, n_chain, n_cold=1, T_cold=1., T_min=1., T_max=1.e15, Ts_in=None, use_inf_final=True):
        """create the temperature ladder object:
            inputs:
                n_chain: scalar integer, total number of parallel tempering chains
                n_cold: scalar integer<=n_chain, total number of T=T_cold chains
                T_cold: scalar float, temperature of 'cold' chain for readout, 1 by default
                T_min: scalar float, minimum temperature of temperature ladder, permitted to be less than T_cold for annealing
                T_max: scalar float, maximum temperature of finite temperature chains
                use_inf_final: scalar boolean, whether the last temperature should be infinity"""
        self.n_chain = n_chain
        self.n_cold = n_cold
        self.T_cold = T_cold
        self.T_min = T_min
        self.Ts_in = Ts_in
        self.T_max = T_max
        self.use_inf_final = use_inf_final
        if Ts_in is not None:
            self.Ts = Ts_in
            self.betas = np.zeros(n_chain)
            for itrj in range(0, n_chain):
                if np.isfinite(Ts_in[itrj]):
                    self.betas[itrj] = 1./Ts_in[itrj]
                else:
                    self.betas[itrj] = 0.

        else:
            self.betas, self.Ts = geometric_spaced_betas(n_chain, n_cold, T_cold, T_min, T_max, use_inf_final=use_inf_final)


def geometric_spaced_betas(n_chain, n_cold, T_cold, T_min, T_max, use_inf_final=True):
    """temeperatures spaced geometrically in a range
        inputs:
            n_chain: scalar integer, total number of chains
            n_cold: scalar integer, number of T=T_cold chains
                    (cold chains are separate from geometric ladder, unless T_cold=T_min
                     in which case 1 of the cold chains is considered part of the geometric ladder)
            T_cold: scalar float, temperature of 'cold' chains
            T_min: scalar float, minimum temperature of geometric ladder
            T_max: scalar float, maximum temperature of finite part of geometric ladder
            use_inf_final: scalar boolean, whether to include an infinite temperature chain separate from the geometric ladder"""
    betas = np.zeros(n_chain)
    Ts = np.zeros(n_chain)
    betas[:n_cold] = 1./T_cold

    if use_inf_final:
        # if T_cold==T_min then include geometric ladder is pinned to n_cold-1 element.
        # otherwise, ladder needs to be pinned to n_cold element, or it will not include an element at T_min
        if T_cold == T_min:
            n_geo = n_chain-n_cold
        else:
            n_geo = n_chain-n_cold-1
        beta_loc = 10**np.linspace(-np.log10(T_min), -np.log10(T_max), n_geo)
        if T_cold == T_min:
            betas[n_cold:n_chain-1] = beta_loc[1:]
        else:
            betas[n_cold:n_chain-1] = beta_loc

        Ts[n_cold:n_chain-1] = 1./betas[n_cold:n_chain-1]

        betas[-1] = 0.
        Ts[-1] = np.inf
    else:
        if T_cold == T_min:
            n_geo = n_chain-n_cold+1
        else:
            n_geo = n_chain-n_cold
        beta_loc = 10**np.linspace(-np.log10(T_min), -np.log10(T_max), n_geo)
        if T_cold == T_min:
            betas[n_cold:n_chain] = beta_loc[1:]
        else:
            betas[n_cold:n_chain] = beta_loc

        Ts[n_cold:] = 1./betas[n_cold:]

    Ts[:n_cold] = T_cold

    return betas, Ts


def entropy_spacing(n_chain_need, betas_in, logLs_in):
    """estimate constant entropy increase spaced chain from an input file of betas and logLs"""
    # as implemented, can't interpolate temperature ladder at non-finite Ts, so remove them
    logLs_in = logLs_in[betas_in > 0.]
    betas_in = betas_in[betas_in > 0.]

    Ts_in = 1./betas_in

    # need to sort the input temperatures and get only unique ones so we can interpolate
    Ts_use = np.unique(Ts_in)
    logLs_use = np.zeros(Ts_use.size)

    for itrf in range(0, Ts_use.size):
        # if there were duplicate temps, average the likelihoods
        logLs_use[itrf] = np.mean(logLs_in[Ts_use[itrf] == Ts_in])

    heat_capacities2 = np.abs(-np.gradient(logLs_use, Ts_use))
    heat_capacity_integ = cumtrapz(heat_capacities2/Ts_use, Ts_use, initial=0.)
    space_heat_need = heat_capacity_integ[Ts_use.size-1]/n_chain_need
    heat_grid_need = np.arange(0, n_chain_need)*space_heat_need
    T_grid_got = 10**InterpolatedUnivariateSpline(heat_capacity_integ, np.log10(Ts_use))(heat_grid_need)

    return T_grid_got


def entropy_spacing_fromfile(n_chain_need, T_file_in, logL_file_in):
    """estimate constant entropy increase spaced chain from an input file of betas and logLs"""
    Ts_in = np.load(T_file_in)
    logLs_in = np.load(logL_file_in)
    betas_in = 1./Ts_in
    betas_in[~np.isfinite(Ts_in)] = 0.
    return entropy_spacing(n_chain_need, betas_in, logLs_in)

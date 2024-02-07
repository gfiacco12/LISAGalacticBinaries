"""C 2023 Matthew C. Digman
module to manage a likelihood object for the galactic binary search"""
import numpy as np
from numba import njit


from DTMCMC.correction_helpers import reflect_cosines, reflect_into_range
from DTMCMC.fisher_manager import set_fishers
from DTMCMC.likelihood import Likelihood

# noah's imports
import WDMWaveletTransforms.wavelet_transforms as wt
import matplotlib.pyplot as plt
import os
import inspect

from Chirp_WDM_funcs import wavemaket_multi_inplace, unpack_wavelets
from instrument_noise import diagonal_dense_log_likelihood_helper, diagonal_dense_snr_helper
import wdm_const as wc
import mcmc_params as mcp
from ra_waveform_time import BinaryTimeWaveformAmpFreqD
import ra_waveform_time as rwt
from graph_helper import get_comp_mass_Mc_Mt

fisher_eps_default = np.array([1.e-25, 1.e-0, 1.e-3, 1.e-6, 1.e-25, 1.e-9, 1.e-9, 1.e-4, 1.e-4, 1.e-4, 1.e-4, 1.e-4])
#eps for alpha = 1.e0, beta = 1.e-3, delta = 1.e-6, 1.e-10, 1.e-19, 1.e-30,

def get_noisy_gb_likelihood(params_fid, noise_AET_dense, sigma_prior_lim, strategy_params):
    """get a likelihood object for a noisy signal
        inputs:
            params_fid: 1D float array, the fiducial parameters
            noise_AET_dense: a noise object from instrument_noise
            sigma_prior_lim: scalar float, how many standard deviations to pad priors by
            strategy_params: DTMCMC.strategy_helpers.StrategyParams object"""
    fwt = BinaryTimeWaveformAmpFreqD(params_fid.copy(), 0, wc.Nt)       # get the waveform object
    signal_full = get_signal_data(params_fid, fwt)                      # get the signal
    noise_realization_full = noise_AET_dense.generate_dense_noise()     # get the noise
    data_full = signal_full+noise_realization_full                      # add the signal and noise
    data_whitened = noise_AET_dense.whiten_dense_data(data_full)        # whiten the signal
    return get_gb_likelihood(params_fid, data_whitened, fwt, noise_AET_dense, sigma_prior_lim, strategy_params)  # return the likelihood object


def get_noiseless_gb_likelihood(params_fid, noise_AET_dense, sigma_prior_lim, strategy_params):
    """get a likelihood object for a noiseless signal
        inputs:
            params_fid: 1D float array, the fiducial parameters
            noise_AET_dense: a noise object from instrument_noise
            sigma_prior_lim: scalar float, how many standard deviations to pad priors by
            strategy_params: DTMCMC.strategy_helpers.StrategyParams object"""
    fwt = BinaryTimeWaveformAmpFreqD(params_fid.copy(), 0, wc.Nt)       # get the waveform object
    signal_full = get_signal_data(params_fid, fwt)                      # get the signal
    signal_whitened = noise_AET_dense.whiten_dense_data(signal_full)    # whiten the signal

    return get_gb_likelihood(params_fid, signal_whitened, fwt, noise_AET_dense, sigma_prior_lim, strategy_params)  # return the likelihood object


def get_gb_likelihood(params_fid, data_use, fwt, noise_AET_dense, sigma_prior_lim, strategy_params):
    """get a likelihood object for galactic binaries given input data"""
    print("getting gb likelihood")
    n_par = params_fid.size
    sigmas_in = np.zeros(n_par)+ strategy_params.sigma_default
    epsilons = fisher_eps_default

    low_lims, high_lims = create_prior_model(params_fid, sigmas_in, sigma_prior_lim)  # generate prior model with default sigmas, will refine later
    # need to create the likelihood object first so we can get the sigmas
    like_obj_temp = GalacticBinaryLikelihood(params_fid, data_use, fwt, noise_AET_dense.inv_chol_SAET, low_lims, high_lims, sigmas_in, epsilons)
    params_fid[:] = like_obj_temp.correct_bounds(params_fid.copy())
    like_obj_temp.params_fid = params_fid
    # get the sigmas we will use to define the prior ranges
    sigma_diag_array, _, _ = set_fishers(np.array([params_fid.copy()]), strategy_params, 1, like_obj_temp)
    sigma_diags = sigma_diag_array[0]
    # sigma_diags[-1] = 1.e-3 * params_fid[-1]

    # get the prior ranges
    low_lims, high_lims = create_prior_model(params_fid, sigma_diags, sigma_prior_lim)

    # reset the relevant parameters of the likelihood object to match the newly computed range
    like_obj_temp.low_lims = low_lims
    like_obj_temp.high_lims = high_lims
    like_obj_temp.sigmas_in = sigma_diags
    #print(sigma_diags)
    return like_obj_temp


class GalacticBinaryLikelihood(Likelihood):
    """class to manage the likelihood-specific essential functions for the sampler"""

    def __init__(self, params_fid, data_use, fwt, inv_chol_SAET, low_lims, high_lims, sigmas_in, epsilons):
        """create the likelihood object for galactic binaries
            inputs:
                params_fid: 1D float array, the fiducial parameters
                data_use: 3D float array, the data to use when calculating the likelihoods
                fwt: an ra_waveform_time.BinaryTimeWaveformAmpFreqD object
                inv_chol_SAET: 3D float array, the inverse cholesky decomposition of the noise
                low_lims: the lower limits of the prior ranges
                high_lims: the upper limits of the prior ranges
                sigmas_in: the standard deviations used to calculate the prior ranges
                epsilons: the shifts to use when calculating the fisher matrices"""
        self.params_fid = params_fid
        self.n_par = params_fid.size
        self.data_use = data_use
        self.fwt = fwt
        self.inv_chol_SAET = inv_chol_SAET
        self.low_lims = low_lims
        self.high_lims = high_lims
        self.sigmas_in = sigmas_in
        self.epsilons = epsilons

        self.waveT = np.zeros((wc.NC, mcp.NMT_max))
        self.Tlists = np.full((wc.NC, mcp.NMT_max), -1, dtype=np.int64)
        self.NUTs = np.zeros(wc.NC, dtype=np.int64)

    def update_internal(self, params_in):
        """internally update the object to match the requested params_in"""
        self.fwt.update_params(params_in)

        NUTs_new = wavemaket_multi_inplace(self.waveT, self.Tlists,
                                           self.fwt.AET_PPTs, self.fwt.AET_FTs, self.fwt.AET_FTds, self.fwt.AET_AmpTs,
                                           wc.NC, wc.Nt, force_nulls=False)
        for itrc in range(0, wc.NC):
            if NUTs_new[itrc] < self.NUTs[itrc]:
                self.waveT[itrc, NUTs_new[itrc]:self.NUTs[itrc]] = 0.
                self.Tlists[itrc, NUTs_new[itrc]:self.NUTs[itrc]] = -1
        self.NUTs = NUTs_new

    def get_loglike(self, params_in):
        """get the log likelihood given a set of parameters params_in"""
        assert self.check_bounds(params_in)
        self.update_internal(params_in)
        if np.sum(self.NUTs) == 0:
            return -np.inf
        return diagonal_dense_log_likelihood_helper(self.inv_chol_SAET, self.Tlists, self.waveT, self.NUTs, self.data_use)

    def get_snr(self, params_in):
        """get the snr given a set of parameters params_in"""
        self.update_internal(params_in)
        return np.sqrt(np.sum(diagonal_dense_snr_helper(self.inv_chol_SAET, self.Tlists, self.waveT, self.NUTs)**2))

    def prior_draw(self):
        """get a draw from the prior"""
        return prior_draw(self.low_lims, self.high_lims)

    def prior_proposal(self, params_in):
        """get a proposal from the prior"""
        params_out = prior_draw(self.low_lims, self.high_lims)
        return params_out, prior_factor(params_in)-prior_factor(params_out), True

    def prior_factor(self, params_in):
        """get the density factor for prior draws, if the prior draws are not uniform"""
        return prior_factor(params_in)

    def correct_bounds(self, params_in):
        """correct the bounds of a draw to be in range, if allowed for this likelihood"""
        return correct_bounds(params_in, self.low_lims, self.high_lims, do_wrap=True)

    def check_bounds(self, params_in):
        """check if the bounds of a draw are in the prior range but do not change them"""
        return check_bounds(params_in, self.low_lims, self.high_lims)


def check_bounds(params_in, low_lims, high_lims):
    """check if a sample is within the prior range"""
    for itrp in range(params_in.size):
        if not low_lims[itrp] < params_in[itrp] < high_lims[itrp]:
            return False
    return True


@njit()
def correct_bounds(params_in, low_lims, high_lims, do_wrap=True):
    """correct the parameters to be in bounds if possible"""

    # before we use the limits, fix the parameters restricted to a cosine to be physical by wrapping phases around their respective poles poles

    # phi is the Ecliptic longitude and is 2pi periodic, it should reflect
    params_in[rwt.idx_costh], params_in[rwt.idx_phi] = reflect_cosines(params_in[rwt.idx_costh], params_in[rwt.idx_phi], np.pi, 2*np.pi)

    params_in[rwt.idx_cosi], params_in[rwt.idx_phi0] = reflect_cosines(params_in[rwt.idx_cosi], params_in[rwt.idx_phi0], np.pi/2, np.pi)

    # psi is polarization angle and is pi periodic
    # note that we could use symmetry with phi0 to further restrict psi to only [0,pi/2)
    params_in[rwt.idx_psi] = (params_in[rwt.idx_psi]) % (np.pi)

    # use symmetry to restrict psi to [0,pi/2)
    # if params_in[rwt.idx_psi]>np.pi/2.:
    #    params_in[rwt.idx_psi] = params_in[rwt.idx_psi] - np.pi/2.
    #    params_in[rwt.idx_phi0] = params_in[rwt.idx_phi0] - np.pi/2.
    # params_in[rwt.idx_phi0] = (params_in[rwt.idx_phi0])%(np.pi)

    # reflect all the parameters into range

    # note that wrapping isn't a safe operation for some jump types like asymmetric jumps, so include an option not to do it
    if do_wrap:
        for itrp in range(0, params_in.size):
            params_in[itrp] = reflect_into_range(params_in[itrp], low_lims[itrp], high_lims[itrp])

    return params_in


@njit()
def prior_draw(low_lims, high_lims, retries=0):
    """do a prior draw"""
    n_par = low_lims.size
    draw = np.zeros(n_par)
    for itrp in range(n_par):
        draw[itrp] = np.random.uniform(low_lims[itrp], high_lims[itrp])
    
    return draw


@njit()
def prior_factor(params_in):
    """get the denstiy factor for prior draws; 0. if draws are uniform"""
    return 0.


def create_prior_model(params_fid, sigmas, sigma_prior_lim):
    """get the limits for the prior model, given fiducial starting parameters,
    an estimate of the standard deviations (only matters for amp0,freq0,freqD currently),
    and how many standard deviations to include (subject to other constraints)
    inputs:
        params_fid: 1D float array, fiducial parameters
        sigmas: standard deviations of the parameters
        sigma_prior_lim: scalar float how many standard deviations to include"""
    n_par = params_fid.size
    low_lims = np.zeros(n_par)
    high_lims = np.zeros(n_par)

    # the amplitude can't be negative but other than that assume we have a coarse estimate available
    low_lims[rwt.idx_amp] = max(params_fid[rwt.idx_amp]-sigma_prior_lim*sigmas[rwt.idx_amp], 0.)
    high_lims[rwt.idx_amp] = params_fid[rwt.idx_amp]+sigma_prior_lim*sigmas[rwt.idx_amp]

    # luminosity distance
    low_lims[rwt.idx_logdl] = max(params_fid[rwt.idx_logdl]-sigma_prior_lim*sigmas[rwt.idx_logdl], 0.)
    high_lims[rwt.idx_logdl] = params_fid[rwt.idx_logdl]+sigma_prior_lim*sigmas[rwt.idx_logdl]

    # the frequency derivative doesn't have any particular hard boundaries (it can be negative in principle) so just do sigma boundaries
    #low_lims[rwt.idx_freqD] = params_fid[rwt.idx_freqD]-2*sigma_prior_lim*sigmas[rwt.idx_freqD]
    #high_lims[rwt.idx_freqD] = params_fid[rwt.idx_freqD]+2*sigma_prior_lim*sigmas[rwt.idx_freqD]
    
    low_lims[rwt.idx_beta] = params_fid[rwt.idx_beta]-5*sigma_prior_lim*sigmas[rwt.idx_beta]
    high_lims[rwt.idx_beta] = params_fid[rwt.idx_beta]+5*sigma_prior_lim*sigmas[rwt.idx_beta]

    # low_lims[rwt.idx_beta] = 1500.
    # high_lims[rwt.idx_beta] = 8000.

    # low_lims[rwt.idx_delta] = 0.7
    # high_lims[rwt.idx_delta] = 4.0
    
    # the frequency second derivative doesn't have any particular hard boundaries (it can be negative in principle) so just do sigma boundaries
    #low_lims[rwt.idx_freqDD] = params_fid[rwt.idx_freqDD]-2*sigma_prior_lim*sigmas[rwt.idx_freqDD]
    #high_lims[rwt.idx_freqDD] = params_fid[rwt.idx_freqDD]+2*sigma_prior_lim*sigmas[rwt.idx_freqDD]
    
    low_lims[rwt.idx_delta] = params_fid[rwt.idx_delta]-10*sigma_prior_lim*sigmas[rwt.idx_delta]
    high_lims[rwt.idx_delta] = params_fid[rwt.idx_delta]+10*sigma_prior_lim*sigmas[rwt.idx_delta]
    
    # make sure initial frequency has at least a few possible characteristic modes at 1/year spacing included
    # but also isn't crossing multiple frequency pixels
    # and if both of those constraints are satisfied then do sigma boundaries
    #delta_freq = max(min(sigma_prior_lim*sigmas[rwt.idx_freq0], 2*wc.DF), 1.25/wc.SECSYEAR)
    #low_lims[rwt.idx_freq0] = max(params_fid[rwt.idx_freq0]-delta_freq, 0.)
    #high_lims[rwt.idx_freq0] = min(params_fid[rwt.idx_freq0]+delta_freq, wc.Nf*wc.DF)

    #delta_alpha = max(params_fid[rwt.idx_alpha]-2*sigma_prior_lim*sigmas[rwt.idx_alpha],0)
    low_lims[rwt.idx_alpha] = params_fid[rwt.idx_alpha]-5*sigma_prior_lim*sigmas[rwt.idx_alpha]
    high_lims[rwt.idx_alpha] = params_fid[rwt.idx_alpha]+2*sigma_prior_lim*sigmas[rwt.idx_alpha]

    # assume ecliptic latitude is just restricted by being physical
    low_lims[rwt.idx_costh] = -1.
    high_lims[rwt.idx_costh] = 1.

    # assume ecliptic longitude is just restricted by being physical and 2pi periodic
    low_lims[rwt.idx_phi] = 0.
    high_lims[rwt.idx_phi] = 2*np.pi

    # assume inclination is just restricted by being physical
    low_lims[rwt.idx_cosi] = -1.
    high_lims[rwt.idx_cosi] = 1.

    # polarization is pi periodic so restrict to 0 and pi
    low_lims[rwt.idx_psi] = 0.
    high_lims[rwt.idx_psi] = np.pi

    # due to degeneracy with polarization we can assume orbital phase is also pi periodic
    low_lims[rwt.idx_phi0] = 0.
    high_lims[rwt.idx_phi0] = np.pi

    #chirp mass priors
    low_lims[rwt.idx_mchirp] = 0.26*wc.MSOLAR
    high_lims[rwt.idx_mchirp] =  1.2*wc.MSOLAR

    #total mass priors
    low_lims[rwt.idx_mtotal] = 0.5*wc.MSOLAR
    high_lims[rwt.idx_mtotal] = 3.0 *wc.MSOLAR

    return low_lims, high_lims


PARAM_LABELS = [r"$\mathcal{A}$", r"$f_0$", r"$f'$", r"$f''$", r"$D_{L}$",r"$M_{T}$ [$M_{\odot}$]", r"$M_{c}$ [$M_{\odot}$]", r"cos$\theta$", r"$\phi$", r"cos$i$", r"$\phi_0$", r"$\psi$", r"$M_{1}$ [$M_{\odot}$]", r"$M_{2}$ [$M_{\odot}$]"] 
PLOT_LABELS = [r"$\mathcal{A}$", r"$\alpha$", r"$\beta$", r"$\delta$", r"$D_{L}$", r"$M_{T}$ [$M_{\odot}$]", r"$M_{c}$ [$M_{\odot}$]", r"cos$\theta$", r"$\phi$", r"cos$i$", r"$\phi_0$", r"$\psi$"] 
#, r"$\kappa$"

def get_param_labels():
    """get just the labels of the parameters"""
    return PARAM_LABELS


def format_samples_output(samples, params_fid, params_to_format = None):
    """take samples and return version suitable for plotting
        inputs:
            samples: a 2D (n_samples,n_par) float array
            params_fid 1D float array, the fiducial parameters"""
    if (params_to_format == None):
        params_to_format = range(0, np.len(samples[0])) # [0, 1, 2, ..., 12]

    samples_got = samples.copy()
    params_fid_got = params_fid.copy()
    labels_loc = PLOT_LABELS.copy()

    labels = []
    samples_fin = []
    params_fin = []

    for i in params_to_format:
        label = labels_loc[i]

        if (i == rwt.idx_amp):
            # get the exponent on the amplitude
            log10_A_base = int(np.floor(np.log10(params_fid[rwt.idx_amp])))
            label = r"$10^{"+str(-log10_A_base)+r"}$"+label
            samples_got[:, rwt.idx_amp] /= 10**log10_A_base             # reduce amplitude to be order unity
            params_fid_got[rwt.idx_amp] /= 10**log10_A_base             # reduce amplitude to be order unity
        elif (i == rwt.idx_logdl):
            samples_got[:, rwt.idx_logdl] = np.log(np.exp(samples_got[:,rwt.idx_logdl])/wc.KPCSEC)    #convert back to kpc
            params_fid_got[rwt.idx_logdl] = np.log(np.exp(params_fid_got[rwt.idx_logdl])/wc.KPCSEC)   #convert back to kpc
        elif (i == rwt.idx_mtotal):
            samples_got[:, rwt.idx_mtotal] /= wc.MSOLAR              # Convert to solar masses
            params_fid_got[rwt.idx_mtotal] /= wc.MSOLAR              # Convert to solar masses
        elif (i == rwt.idx_mchirp):
            samples_got[:, rwt.idx_mchirp] /= wc.MSOLAR
            params_fid_got[rwt.idx_mchirp] /= wc.MSOLAR
        
        labels.append(label)
        params_fin.append(params_fid_got[i])

    for sample in samples_got:
        formatted_sample = []
        for i in params_to_format:
            formatted_sample.append(sample[i])

        m1, m2, q = get_comp_mass_Mc_Mt(sample[rwt.idx_mchirp], sample[rwt.idx_mtotal])
        formatted_sample.append(m1)
        formatted_sample.append(m2)
        formatted_sample.append(q)

        # gamma = s[rwt.idx_freqDD] 
        # alpha = s[rwt.idx_freq0]
        # beta = s[rwt.idx_freqD]
        # delta = (gamma - (11/3) * (beta**2 / alpha))
        # s.append(delta)
        samples_fin.append(formatted_sample)

    m1_params = 0.8
    m2_params = 0.6
    q_params = 0.75
    params_fin.append(m1_params)
    params_fin.append(m2_params)
    params_fin.append(q_params)

    labels.append(r"$M_{1}$")
    labels.append(r"$M_{2}$")
    labels.append(r"$q$")
    print (labels)
    
    return np.array(samples_fin), np.array(params_fin), labels


def get_signal_data(params_true, fwt):
    """get the wavelet domain data for a signal with the specified parameters
        inputs:
            params_true: a 1D float array of parameters
            fwt: a ra_waveform_time.BinaryTimeWaveformAmpFreqD object"""

    # max number of wavelet layers times the number of time pixels
    fwt.update_params(params_true)

    waveT = np.zeros((wc.NC, mcp.NMT_max))
    Tlists = np.full((wc.NC, mcp.NMT_max), -1, dtype=np.int64)
    NUTs = np.zeros(wc.NC, dtype=np.int64)

    NUTs_new = wavemaket_multi_inplace(waveT, Tlists, fwt.AET_PPTs, fwt.AET_FTs, fwt.AET_FTds, fwt.AET_AmpTs, wc.NC, wc.Nt, force_nulls=False)
    for itrc in range(0, wc.NC):
        if NUTs_new[itrc] < NUTs[itrc]:
            waveT[itrc, NUTs_new[itrc]:NUTs[itrc]] = 0.
            Tlists[itrc, NUTs_new[itrc]:NUTs[itrc]] = -1
    NUTs = NUTs_new

    signal_full = unpack_wavelets(wc.NC, waveT, Tlists, NUTs)
    return signal_full

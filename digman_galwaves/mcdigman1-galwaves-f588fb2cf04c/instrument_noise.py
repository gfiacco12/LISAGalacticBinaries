"""C 2023 Matthew C. Digman
get the instrument noise profile"""

import numpy as np
from numpy.random import normal
from numba import njit

from WDMWaveletTransforms.transform_freq_funcs import phitilde_vec
import wdm_const as wc
from galactic_fit_helpers import SAE_from_file


def get_galactic_novar_noise_model():
    """get a model for the instrument and galactic noise spectrum with no time variability
    automatically assumes the constants in wdm_const"""
    SAET_m_temp = instrument_noise_AET_wdm_m()
    SAET_m = SAET_m_temp.copy()
    SAE_fit = SAE_from_file(np.int64(np.round(wc.Tobs/wc.SECSYEAR)), SAET_m_temp[:, 0])
    SAET_m[:, 0] = SAE_fit
    SAET_m[:, 1] = SAE_fit

    SAET_cur = np.zeros((wc.Nt, wc.Nf, wc.NC))
    for itrc in range(0, wc.NC):
        SAET_cur[:, :, itrc] = SAET_m[:, itrc]

    noise_AET_dense = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_cur, prune=True)
    return noise_AET_dense


def instrument_noise_LDC(f):
    """Power spectral density of the detector noise and transfer frequency, to match to LDC sangria data"""
    SAE = np.zeros(f.size)
    # Sps = 2.25e-22*0.045
    Sps = 9.e-24  # should match sangria v2? Should it be backlinknoise or readoutnoise?
    # Sacc = 9.0e-30
    Sacc = 5.76e-30  # from sangria v2
    fonfs = f/wc.fstr
    # To match the LDC power spectra need a factor of 2 here. No idea why... (one sided/two sided?)
    LC = 2.0*fonfs*fonfs
    # roll-offs
    rolla = (1.0+pow((4.0e-4/f), 2.0))*(1.0+pow((f/8.0e-3), 4.0))
    rollw = (1.0+pow((2.0e-3/f), 4.0))
    # Calculate the power spectral density of the detector noise at the given frequency
    # not and exact match to the LDC, but within 10%
    SAE = LC*16.0/3.0*pow(np.sin(fonfs), 2.0)\
            * ((2.0+np.cos(fonfs))*(Sps)*rollw
               + 2.0*(3.0+2.0*np.cos(fonfs)+np.cos(2.0*fonfs))*(Sacc/pow(2.0*np.pi*f, 4.0)*rolla)) / pow(2.0*wc.Larm, 2.0)
    return SAE


def instrument_noise_AET(f):
    """get power spectral density in all 3 channels, assuming identical in all arms"""
    # see arXiv:2005.03610
    # see arXiv:1002.1291
    fonfs = f/wc.fstr

    LC = 64/(3*wc.Larm**2)
    mult_all = LC*fonfs**2*np.sin(fonfs)**2
    mult_sa = (4*wc.Sacc/(2*np.pi)**4)*(1+16.e-8/f**2)*(1.0+(f/8.0e-3)**4.)/f**4
    mult_sp = wc.Sps*(1.0+(2.0e-3/f)**4.)

    cosfonfs = np.cos(fonfs)

    SAET = np.zeros((f.size, wc.NC))

    # SAET[:,0] = mult_all*(mult_sa*(1+cosfonfs+cosfonfs**2)+mult_sp*(2+cosfonfs))
    SAET[:, 0] = instrument_noise_LDC(f)  # TODO make this all self consistent
    SAET[:, 1] = SAET[:, 0]
    SAET[:, 2] = mult_all*(mult_sa/2*(1-2*cosfonfs+cosfonfs**2)+mult_sp*(1-cosfonfs))
    return SAET


def instrument_noise_AET_wdm_m():
    """get the instrument noise curve as a function of frequency for the wdm wavelet decomposition
    if prune=True, cut the 1st and last values, which may not bet calculated correctly"""

    # TODO why no plus 1?
    ls = np.arange(-wc.Nt//2, wc.Nt//2)
    fs = ls/wc.Tobs
    phif = np.sqrt(wc.dt)*phitilde_vec(2*np.pi*wc.dt*fs, wc.Nf, wc.nx)
    # TODO check ad hoc normalization factor
    SAET_m = instrument_noise_AET_wdm_loop(phif)
    return SAET_m


def instrument_noise_AET_wdm_loop(phif):
    """helper to get the instrument noise for wdm"""
    # realistically this really only needs run once and is fast enough without jit
    # TODO check normalization
    # TODO get first and last bins correct
    # nrm =   np.sqrt(2*wc.Nf*wc.dt)*np.linalg.norm(phif)
    # nrm =   2*np.sqrt(2*wc.dt)*np.linalg.norm(phif)
    nrm = np.sqrt(12318/wc.Nf)*np.linalg.norm(phif)
    print('nrm instrument', nrm)
    phif /= nrm
    phif2 = phif**2

    SAET_M = np.zeros((wc.Nf, wc.NC))
    half_Nt = wc.Nt//2
    fs_long = np.arange(-half_Nt, half_Nt+wc.Nf*half_Nt)/wc.Tobs
    # prevent division by 0
    fs_long[half_Nt] = fs_long[half_Nt+1]
    SAET_long = instrument_noise_AET(fs_long)
    # excise the f=0 point
    SAET_long[half_Nt, :] = 0.
    # apply window in loop
    for m in range(0, wc.Nf):
        SAET_M[m] = np.dot(phif2, SAET_long[m*half_Nt:(m+2)*half_Nt])

    return SAET_M


class DiagonalNonstationaryDenseInstrumentNoiseModel:
    """a class to handle the fully diagonal stationary
    instrument noise model to feed to snr and fisher matrix calculations"""

    def __init__(self, SAET, prune):
        """initialize the instrument noise model
        inputs:
            SAET: 3D float array, the noise model
            prune: scalar boolean, whether to ignore last bin"""
        self.prune = prune
        self.SAET = SAET
        self.inv_SAET = np.zeros((wc.Nt, wc.Nf, wc.NC))
        self.inv_chol_SAET = np.zeros((wc.Nt, wc.Nf, wc.NC))
        self.chol_SAET = np.zeros((wc.Nt, wc.Nf, wc.NC))
        for j in range(0, wc.Nt):
            for itrc in range(0, wc.NC):
                self.chol_SAET[j, :, itrc] = np.sqrt(self.SAET[j, :, itrc])
                self.inv_chol_SAET[j, :, itrc] = 1./self.chol_SAET[j, :, itrc]
                self.inv_SAET[j, :, itrc] = self.inv_chol_SAET[j, :, itrc]**2

        self.mean_SAE = np.mean(np.mean(self.SAET[:, :, 0:2], axis=0), axis=1)
        self.inv_chol_mean_SAE = 1./np.sqrt(self.mean_SAE)

        if self.prune:
            self.chol_SAET[:, 0, :] = 0.
            self.inv_chol_SAET[:, 0, :] = 0.
            self.inv_SAET[:, 0, :] = 0.
            self.mean_SAE[0] = 0.
            self.inv_chol_mean_SAE[0] = 0.

    def whiten_dense_data(self, wavelet_data):
        """wrapper for dense data whitening"""
        return diagonal_dense_whiten_dense_data_helper(self.inv_chol_SAET, wavelet_data)

    def get_log_likelihood(self, waveform_sig, wavelet_data_whitened):
        """calculate reduced log likelihood"""
        pixel_lists, wave_coeffs, NUs = waveform_sig.get_unsorted_coeffs()
        return diagonal_dense_log_likelihood_helper(self.inv_chol_SAET, pixel_lists, wave_coeffs, NUs, wavelet_data_whitened)

    def generate_dense_noise(self):
        """generate random noise for full matrix"""
        noise_res = np.zeros((wc.NC, wc.Nt, wc.Nf))
        for j in range(0, wc.Nt):
            noise_res[:, j, :] = normal(0., 1., (wc.NC, wc.Nf))*self.chol_SAET[j, :, :].T
        return noise_res

    def get_sparse_snrs(self, NUs, pixel_lists, wave_coeffs):
        """get snr of waveform in each channel"""
        return diagonal_dense_snr_helper(self.inv_chol_SAET, pixel_lists, wave_coeffs, NUs)


@njit()
def diagonal_dense_whiten_dense_data_helper(inv_chol_SAET, wavelet_data):
    """whiten data with noise model"""
    data_whitened = np.zeros((wc.NC, wc.Nt, wc.Nf))
    for itrc in range(0, wc.NC):
        data_whitened[itrc] = inv_chol_SAET[:, :, itrc]*wavelet_data[itrc]
    return data_whitened


@njit()
def diagonal_dense_snr_helper(inv_chol_SAET, pixel_lists, wave_coeffs, NUs):
    """helper to get the snr of the waveform in each channel
    inputs:
        inv_chol_SAET: a 3D float array, the inverse cholesky decomposition of the noise
        pixel_lists: a 2D integer array, the indexes of active pixels (end with first -1)
        wave_coeffs: a 2D float array, the sparse wavelet coefficients
        NUs: a 1D integer array the number of active pixels in each channel
    ouputs:
        snrs: a 1D float array of the snr in each channel"""
    snr2s = np.zeros(wc.NC)
    for itrc in range(0, wc.NC):
        i_itrs = np.mod(pixel_lists[itrc, 0:NUs[itrc]], wc.Nf).astype(np.int64)
        j_itrs = (pixel_lists[itrc, 0:NUs[itrc]]-i_itrs)//wc.Nf
        for mm in range(0, NUs[itrc]):
            mult = inv_chol_SAET[j_itrs[mm], i_itrs[mm], itrc]*wave_coeffs[itrc, mm]
            snr2s[itrc] += mult*mult
    return np.sqrt(snr2s)


# @njit()
def diagonal_dense_log_likelihood_helper(inv_chol_SAET, pixel_lists, wave_coeffs, NUs, wavelet_data_whitened):
    """calculate the reduced log likelihood for a wavelet object given input data matrix and a diagonal stationary input noise model
        inputs:
            inv_chol_SAET: a 3D float array, the inverse cholesky decomposition of the noise
            pixel_lists: a 2D integer array, the indexes of active pixels (end with first -1)
            wave_coeffs: a 2D float array, the sparse wavelet coefficients
            NUs: a 1D integer array the number of active pixels in each channel
            wavelet_data_whitened: a 3D float array, the whitened (non-sparse) data in the wavelet domain
        outputs:
            log_likelihood: scalar float log likelihood"""
    # this uses a helper because it is important to perforamce and currently this method works faster than using jitclass
    res = 0.
    for itrc in range(0, wc.NC):
        for itrp in range(0, NUs[itrc]):
            i = np.int64(np.mod(pixel_lists[itrc, itrp], wc.Nf))
            j = (pixel_lists[itrc, itrp]-i)//wc.Nf
            inv_chol_Snm = inv_chol_SAET[j, i, itrc]
            coeff_loc = inv_chol_Snm*wave_coeffs[itrc, itrp]
            data_loc = wavelet_data_whitened[itrc, j, i]
            signal_part = -coeff_loc**2/2
            data_part = data_loc*coeff_loc
            res += data_part+signal_part
    return res

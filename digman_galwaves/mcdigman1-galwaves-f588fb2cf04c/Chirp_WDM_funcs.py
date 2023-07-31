"""C 2023 Matthew C. Digman
helper functions for Chirp_WDM"""
from numba import njit
import numpy as np

import wdm_const as wc
from mcmc_params import evcTs, evsTs, NfsamT


@njit()
def unpack_wavelets(NC, wave_in, lists_in, NUs):
    """helper to unpack a NC channel wavelet decomposition to array
        inputs:
            NC: scalar integer, number of channels
            wave_in: 2D float array, sparse wavelet coefficients
            lists_in: 2D integer array, indexes of sparse wavelet coefficients
            NUs: 1D integer array, number of sparse coefficients in each channel
        outputs:
           wave: 3D float array, dense wavelet coefficients """
    # initialize the array
    wave = np.zeros((NC, wc.Nt, wc.Nf))
    # unpack the signal
    for itrc in range(0, NC):
        for mm in range(0, NUs[itrc]):
            i = lists_in[itrc, mm] % wc.Nf
            j = (lists_in[itrc, mm]-i)//wc.Nf
            wave[itrc, j, i] = wave_in[itrc, mm]
    return wave


@njit()
def wavemaket_multi_inplace(waveTs, Tlists, Phases, fas, fdas, Amps, NC, nt_range, force_nulls=False):
    """compute the actual wavelets using taylor time method
        inputs:
            waveTs: 2D float array of sparse wavelet coefficients to write to
            Tlists: 2D int array of sparse wavelet coefficient indexes to write to
            Phases: 2D float array of phases
            fas: 2D float array of frequencies
            fdas: 2D float array of frequency derivatives
            Amps: 2D float array of amplitudes
            NC: number of channels
            nt_range: maximum time index to go to
            force_nulls: whether to impute zeros for wavelet coefficients where nulls make getting the actual values difficult
        outputs:
            NUs: 1D integer array of number of coefficients in each channel
            note that waveTs and Tlists are also functionally output, because they are written tos"""
    NUs = np.zeros(NC, dtype=np.int64)
    for itrc in range(0, NC):
        mm = 0
        for j in range(0, nt_range):
            j_ind = j

            y0 = fdas[itrc, j]/wc.dfd
            ny = np.int64(np.floor(y0))
            n_ind = ny+wc.Nfd_negative

            if 0 <= n_ind < wc.Nfd-2:
                c = Amps[itrc, j]*np.cos(Phases[itrc, j])
                s = Amps[itrc, j]*np.sin(Phases[itrc, j])

                dy = y0-ny
                fa = fas[itrc, j]
                za = fa/wc.df
                Nfsam1_loc = NfsamT[n_ind]
                Nfsam2_loc = NfsamT[n_ind+1]
                HBW = (min(Nfsam1_loc, Nfsam2_loc)-1)*wc.df/2

                # lowest frequency layer
                kmin = max(0, np.int64(np.ceil((fa-HBW)/wc.DF)))

                # highest frequency layer
                kmax = min(wc.Nf-1, np.int64(np.floor((fa+HBW)/wc.DF)))
                for k in range(kmin, kmax+1):
                    Tlists[itrc, mm] = j_ind*wc.Nf+k

                    zmid = (wc.DF/wc.df)*k

                    kk = np.floor(za-zmid-0.5)
                    zsam = zmid+kk+0.5
                    kk = np.int64(kk)
                    dx = za-zsam  # used for linear interpolation

                    # interpolate over frequency
                    jj1 = kk+Nfsam1_loc//2
                    jj2 = kk+Nfsam2_loc//2

                    assert evcTs[n_ind, jj1] != 0.
                    assert evcTs[n_ind, jj1+1] != 0.
                    assert evcTs[n_ind+1, jj2] != 0.
                    assert evcTs[n_ind+1, jj2+1] != 0.

                    y = (1.-dx)*evcTs[n_ind, jj1]+dx*evcTs[n_ind, jj1+1]
                    yy = (1.-dx)*evcTs[n_ind+1, jj2]+dx*evcTs[n_ind+1, jj2+1]

                    z = (1.-dx)*evsTs[n_ind, jj1]+dx*evsTs[n_ind, jj1+1]
                    zz = (1.-dx)*evsTs[n_ind+1, jj2]+dx*evsTs[n_ind+1, jj2+1]

                    # interpolate over fdot
                    y = (1.-dy)*y+dy*yy
                    z = (1.-dy)*z+dy*zz

                    if (j_ind+k) % 2:
                        waveTs[itrc, mm] = -(c*z+s*y)
                    else:
                        waveTs[itrc, mm] = (c*y-s*z)

                    mm += 1
                    # end loop over frequency layers
            elif force_nulls:
                # we know what the indices would be for values not precomputed in the table
                # so force values outside the range of the table to 0 instead of dropping, in order to get likelihoods right
                # which is particularly important around total nulls which can be quite constraining on the parameters
                # but have large spikes in frequency derivative
                # note that if this happens only very rarely, we could also just actually calculate the non-precomputed coefficient
                fa = fas[itrc, j]

                Nfsam1_loc = np.int64((wc.BW+wc.dfd*wc.Tw*ny)/wc.df)
                if Nfsam1_loc % 2 == 1:
                    Nfsam1_loc += 1

                HBW = (Nfsam1_loc-1)*wc.df/2
                # lowest frequency layer
                kmin = max(0, np.int64(np.ceil((fa-HBW)/wc.DF)))

                # highest frequency layer
                kmax = min(wc.Nf-1, np.int64(np.floor((fa+HBW)/wc.DF)))

                for k in range(kmin, kmax+1):
                    Tlists[itrc, mm] = j_ind*wc.Nf+k
                    waveTs[itrc, mm] = 0.
                    mm += 1

        NUs[itrc] = mm

    return NUs

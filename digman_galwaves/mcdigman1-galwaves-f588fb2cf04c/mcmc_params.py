"""C 2023 Matthew C. Digman
store parameters that will be globally constant for entire mcmc run"""
import numpy as np

from coefficientsWDM_time_helpers import get_evTs
import wdm_const as wc

# interpolation for wavelet taylor expansion
NfsamT, evcTs, evsTs = get_evTs(check_cache=True, hf_out=False)

# maximum number of wavelet coefficients needed by taylor expansion
fds = wc.dfd*np.arange(-wc.Nfd_negative, wc.Nfd-wc.Nfd_negative)
NMT_max = np.int64(np.ceil((wc.BW+fds[wc.Nfd-1]*wc.Tw)/wc.DF))*wc.Nt

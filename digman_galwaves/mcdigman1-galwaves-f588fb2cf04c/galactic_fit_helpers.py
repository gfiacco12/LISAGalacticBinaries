"""C 2023 Matthew C. Digman
scratch to test processing of galactic background"""
import numpy as np
import wdm_const as wc


def SAE_gal_model(f, log10A, log10f2, log10f1, log10fknee, alpha):
    """model from arXiv:2103.14598 for galactic binary confusion noise amplitude"""
    return 10**log10A/2*f**(5/3)*np.exp(-(f/10**log10f1)**alpha)*(1+np.tanh((10**log10fknee-f)/10**log10f2))


def SAE_from_res(res, TobsYEAR, SAE_m):
    """retrieve the appropriate SAE given model parameters"""
    SAE_got = SAE_m.copy()
    fs = np.arange(0, wc.Nf)*wc.DF
    arg_cut = wc.Nf-1
    fit_mask = (fs > 1.e-5) & (fs < fs[arg_cut])
    a1 = res[0]
    ak = res[1]
    b1 = res[2]
    bk = res[3]
    log10A = res[4]
    log10f2 = res[5]
    alpha = res[6]
    log10f1 = (a1*np.log10(TobsYEAR)+b1)
    log10fknee = (ak*np.log10(TobsYEAR)+bk)
    SAE_got[fit_mask] += SAE_gal_model(fs[fit_mask], log10A, log10f2, log10f1, log10fknee, alpha)
    return SAE_got


def SAE_from_file(TobsYEAR, SAE_m, gal_file='digman_galwaves/mcdigman1-galwaves-f588fb2cf04c/fit_gal_best_res.npy'):
    """get appropriate SAE given model parameters in a file"""
    res = np.load(gal_file)
    return SAE_from_res(res, TobsYEAR, SAE_m)

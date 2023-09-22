"""C 2023 Matthew C. Digman
subroutines for running LISA code"""
import numpy as np
from numba import njit

from ra_waveform_freq import RAantenna_inplace, spacecraft_vec, get_xis_inplace, get_tensor_basis

import wdm_const as wc
from algebra_tools import gradient_homog_2d_inplace
from scipy.optimize import fsolve

idx_amp = 0
idx_freq0 = 1
idx_freqD = 2
idx_freqDD = 3
idx_logdl = 4
idx_mtotal = 5
idx_mchirp = 6
idx_costh = 7
idx_phi = 8
idx_cosi = 9
idx_phi0 = 10
idx_psi = 11
idx_mass1 = 12
idx_mass2 = 13
idx_iwd = 14
idx_freqDDD = 15

# TODO do consistency checks
class BinaryTimeWaveformAmpFreqD():
    """class to store a binary waveform in time domain and update for search
        assuming input binary format based on amplitude, frequency, and frequency derivative"""

    def __init__(self, params, NT_min, NT_max):
        """initalize the object
            inputs:
                params: 1D float array, the starting parameters
                NT_min: scalar integer, the lowest time pixel to use
                NT_max: scalar integer, the highest time pixel to use"""
        self.params = params
        self.NT_min = NT_min
        self.NT_max = NT_max
        self.NT = self.NT_max-self.NT_min

        self.nt_low = self.NT_min
        self.nt_high = self.NT_max

        self.nt_range = self.nt_high-self.nt_low

        self.TTs = wc.DT*np.arange(self.nt_low, self.nt_high)

        self.AmpTs = np.zeros(self.NT)
        self.PPTs = np.zeros(self.NT)
        self.FTs = np.zeros(self.NT)
        self.FTds = np.zeros(self.NT)
        self.FTdds = np.zeros(self.NT)
        self.FTddds = np.zeros(self.NT)

        self.dRRs = np.zeros((wc.NC, self.NT))
        self.dIIs = np.zeros((wc.NC, self.NT))
        self.RRs = np.zeros((wc.NC, self.NT))
        self.IIs = np.zeros((wc.NC, self.NT))

        self.xas = np.zeros(self.NT)
        self.yas = np.zeros(self.NT)
        self.zas = np.zeros(self.NT)
        self.xis = np.zeros(self.NT)
        self.kdotx = np.zeros(self.NT)

        _, _, _, self.xas[:], self.yas[:], self.zas[:] = spacecraft_vec(self.TTs)

        self.AET_AmpTs = np.zeros((wc.NC, self.NT))
        self.AET_PPTs = np.zeros((wc.NC, self.NT))
        self.AET_FTs = np.zeros((wc.NC, self.NT))
        self.AET_FTds = np.zeros((wc.NC, self.NT))

        self.update_params(params)

    def update_params(self, params):
        """update the parameters of the waveform object
            inputs:
                params: 1D float array of new parameters"""
        self.params = params
        self.update_intrinsic()
        self.update_extrinsic()

    def update_intrinsic(self):
        """get amplitude and phase for the waveform model"""
        # amp = np.sqrt(wc.Tobs/(8*wc.Nt*wc.Nf))*self.params[0]
        amp = self.params[idx_amp]
        costh = self.params[idx_costh]  # np.cos(np.pi/2-self.params[idx_theta] )
        phi = self.params[idx_phi]
        freq0 = self.params[idx_freq0]
        freqD = self.params[idx_freqD]
        freqDD = self.params[idx_freqDD]
        freqDDD = self.params[idx_freqDDD]
        # cosi = self.params[idx_cosi]#np.cos(self.params[idx_incl])
        phi0 = self.params[idx_phi0]  # +np.pi
        # psi = self.params[idx_psi]
        dl = np.exp(self.params[idx_logdl])
        m_total = self.params[idx_mtotal]
        m_chirp = self.params[idx_mchirp]
        mass1 = self.params[idx_mass1]
        mass2 = self.params[idx_mass2]
        I_wd = self.params[idx_iwd]

        #physical model constants - 1PN
        # eta = (m_chirp/m_total)**(5/3)
        # fdot_pp = 96/5*np.pi**(8/3)*freq0**(11/3)*m_chirp**(5/3)
        # freqD_1PN = 96/5*np.pi**(8/3)*freq0**(11/3)*m_chirp**(5/3) * (1 + ((743/1344)-(11*eta/16))*(8*np.pi*m_total*freq0)**(2/3))
        # freqDD_1PN = 96/5*np.pi**(8/3)*freq0**(8/3)*m_chirp**(5/3)*freqD_1PN * ((11/3) + (13/3)*((743/1344)-(11*eta/16))*(8*np.pi*m_total*freq0)**(2/3))
        # freqDDD_1PN = (19/3) * ((freqD_1PN * freqDD_1PN) / freq0) * ( 1 + (2/19) * (fdot_pp / freqD_1PN) * (1 + ((13/3)*(freqD_1PN**2 / (freq0 * freqDD_1PN)))) * (((743/1344)-(11*eta/16))*((8*np.pi*m_total*freq0)**(2/3))) )
        # amp_1PN = np.pi**(2/3) * m_chirp**(5/3) * freq0**(2/3) / dl 

        # #physical model - tides
        # #I_wd = 8.51e-10 * ( (mass1/(0.6*wc.MSOLAR))**(1/3) + (mass2/(0.6*wc.MSOLAR))**(1/3) )
        # chirpMass = (mass1*mass2)**(3/5) / (mass1 + mass2)**(1/5)
        # totalMass = mass1 + mass2
        # freqD_tides = 96/5*np.pi**(8/3)*freq0**(11/3)*chirpMass**(5/3) * (1 + ((3*I_wd*(np.pi*freq0)**(4/3)/chirpMass**(5/3)) / (1 - (3*I_wd*(np.pi*freq0)**(4/3)/chirpMass**(5/3)))) )
        # freqDD_tides = 96/5*np.pi**(8/3)*freq0**(11/3)*chirpMass**(5/3) * (freqD_tides/freq0) * ( ((11/3) - (7*I_wd*(np.pi*freq0)**(4/3) / chirpMass**(5/3))) / ((1 - (3*I_wd*(np.pi*freq0)**(4/3)/chirpMass**(5/3)))**2))
        # amp_tides = np.pi**(2/3) * chirpMass**(5/3) * freq0**(2/3) / dl

        # #physical model - tides, Moment of Inertia, Chirp Mass
        # freqD_tides_Iwd = 96/5*np.pi**(8/3)*freq0**(11/3)*m_chirp**(5/3) * (1 + ((3*I_wd*(np.pi*freq0)**(4/3)/m_chirp**(5/3)) / (1 - (3*I_wd*(np.pi*freq0)**(4/3)/m_chirp**(5/3)))) )
        # freqDD_tides_Iwd = 96/5*np.pi**(8/3)*freq0**(11/3)*m_chirp**(5/3) * (freqD_tides_Iwd/freq0) * ( ((11/3) - (7*I_wd*(np.pi*freq0)**(4/3) / m_chirp**(5/3))) / ((1 - (3*I_wd*(np.pi*freq0)**(4/3)/m_chirp**(5/3)))**2))

        # #reference times for each model
        # TTRef = TaylorT3_ref_time_match(m_total, m_chirp, freq0, TaylorF2_ref_time_guess(m_total,m_chirp,freq0))
        # TTRef_tides = TaylorT3_ref_time_match(totalMass, chirpMass, freq0, TaylorF2_ref_time_guess(totalMass,chirpMass,freq0))
        # TTRef_Iwd = TaylorT3_ref_time_match(1.2*wc.MSOLAR, m_chirp, freq0, TaylorF2_ref_time_guess(1.2*wc.MSOLAR,m_chirp,freq0))

        kv, _, _ = get_tensor_basis(phi, costh)  # TODO check intrinsic extrinsic separation here
        get_xis_inplace(kv, self.TTs, self.xas, self.yas, self.zas, self.xis)
        AmpFreqDeriv_inplace(self.AmpTs, self.PPTs, self.FTs, self.FTds, self.FTdds, self.FTddds ,amp, phi0, freq0, freqD, freqDD, freqDDD, 0, self.xis, self.TTs.size)
        #AmpFreqDeriv_inplace(self.AmpTs, self.PPTs, self.FTs, self.FTds, self.FTdds, self.FTddds, amp_1PN, phi0, freq0, freqD_1PN, freqDD_1PN, freqDDD_1PN,TTRef, self.xis, self.TTs.size)
        #AmpFreqDeriv_inplace(self.AmpTs, self.PPTs, self.FTs, self.FTds, self.FTdds, self.FTddds,amp_tides, phi0, freq0, freqD_tides, freqDD_tides,freqDDD_tides, TTRef_tides, self.xis, self.TTs.size)
        #AmpFreqDeriv_inplace(self.AmpTs, self.PPTs, self.FTs, self.FTds, self.FTdds, self.FTddds, amp_1PN, phi0, freq0, freqD_tides_Iwd, freqDD_tides_Iwd, freqDDD_tides_Iwd,TTRef_Iwd, self.xis, self.TTs.size)
#
    def update_extrinsic(self):
        """update the internal state for the extrinsic parts of the parameters"""
        # Calculate cos and sin of sky position, inclination, polarization

        # amp = self.params[idx_amp]
        costh = self.params[idx_costh]  # np.cos(np.pi/2-self.params[idx_theta] )
        phi = self.params[idx_phi]
        # freq0 = self.params[idx_freq0]
        # freqD = self.params[idx_freqD]
        cosi = self.params[idx_cosi]  # np.cos(self.params[idx_incl])
        # phi0 = self.params[idx_phi0]#+np.pi
        psi = self.params[idx_psi]
        RAantenna_inplace(self.RRs, self.IIs, cosi, psi, phi, costh, self.TTs, self.FTs, 0, self.NT, self.kdotx)  # TODO fix F_min and nf_range
        ExtractAmpPhase_inplace(self.AET_AmpTs, self.AET_PPTs, self.AET_FTs, self.AET_FTds,
                                self.AmpTs, self.PPTs, self.FTs, self.FTds, self.RRs, self.IIs, self.dRRs, self.dIIs, self.NT)

def TruthParamsCalculator(freq0, mass1, mass2, dl):
    #calculate frequencies using physical models and input them as truth params for the code
    I_wd = 8.51e-10 * ( (mass1/(0.6*wc.MSOLAR))**(1/3) + (mass2/(0.6*wc.MSOLAR))**(1/3) ) # this is in s^3
    chirpMass = (mass1*mass2)**(3/5) / (mass1 + mass2)**(1/5)
    totalMass = mass1 + mass2
    eta = (chirpMass/totalMass)**(5/3)
    fdot_pp = 96/5*np.pi**(8/3)*freq0**(11/3)*chirpMass**(5/3)
    #tides
    fdot_tides = 96/5*np.pi**(8/3)*freq0**(11/3)*chirpMass**(5/3) * (1 + ((3*I_wd*(np.pi*freq0)**(4/3)/chirpMass**(5/3)) / (1 - (3*I_wd*(np.pi*freq0)**(4/3)/chirpMass**(5/3)))) )
    fddot_tides = 96/5*np.pi**(8/3)*freq0**(11/3)*chirpMass**(5/3) * (fdot_tides/freq0) * ( ((11/3) - (7*I_wd*(np.pi*freq0)**(4/3) / chirpMass**(5/3))) / ((1 - (3*I_wd*(np.pi*freq0)**(4/3)/chirpMass**(5/3)))**2))
    #1PN
    fdot = 96/5*np.pi**(8/3)*freq0**(11/3)*chirpMass**(5/3) * (1 + ((743/1344)-(11*eta/16))*(8*np.pi*totalMass*freq0)**(2/3))
    fddot = 96/5*np.pi**(8/3)*freq0**(8/3)*chirpMass**(5/3)*(fdot) * ((11/3) + (13/3)*((743/1344)-(11*eta/16))*(8*np.pi*totalMass*freq0)**(2/3))
    fdddot = (19/3) * ((fdot * fddot) / freq0) * ( 1 + ((2/19) * (fdot_pp / fdot) * (1 + ((13/3)*(fdot**2 / (freq0 * fddot)))) * (((743/1344)-(11*eta/16))*((8*np.pi*totalMass*freq0)**(2/3)))) )

    amp = np.pi**2/3 * chirpMass**(5/3) * freq0**(2/3) / dl

    return fdot, fddot, fdot_tides, fddot_tides, amp, I_wd, fdddot

@njit()
def ExtractAmpPhase_inplace(AET_Amps, AET_Phases, AET_FTs, AET_FTds, AA, PP, FT, FTd, RRs, IIs, dRRs, dIIs, NT):
    """get the amplitude and phase for LISA"""
    # TODO check absolute phase aligns with Extrinsic_inplace
    polds = np.zeros(wc.NC)
    js = np.zeros(wc.NC)

    gradient_homog_2d_inplace(RRs, wc.DT, NT, 3, dRRs)
    gradient_homog_2d_inplace(IIs, wc.DT, NT, 3, dIIs)

    n = 0
    for itrc in range(0, wc.NC):
        polds[itrc] = np.arctan2(IIs[itrc, n], RRs[itrc, n])
        if polds[itrc] < 0.:
            polds[itrc] += 2*np.pi
    for n in range(0, NT):
        fonfs = FT[n]/wc.fstr

        # including TDI + fractional frequency modifiers
        Ampx = AA[n]*(8*fonfs*np.sin(fonfs))
        Phase = PP[n]
        for itrc in range(0, wc.NC):
            RR = RRs[itrc, n]
            II = IIs[itrc, n]

            if RR == 0. and II == 0.:
                # TODO is this the correct way to handle both ps being 0?
                p = 0.
                AET_FTs[itrc, n] = FT[n]
            else:
                p = np.arctan2(II, RR)
                AET_FTs[itrc, n] = FT[n]-(II*dRRs[itrc, n]-RR*dIIs[itrc, n])/(RR**2+II**2)/(2*np.pi)

            if p < 0.:
                p += 2*np.pi

            # TODO implement integral tracking of js
            if p-polds[itrc] > 6.:
                js[itrc] -= 2*np.pi
            if polds[itrc]-p > 6.:
                js[itrc] += 2*np.pi
            polds[itrc] = p

            AET_Amps[itrc, n] = Ampx*np.sqrt(RR**2+II**2)
            AET_Phases[itrc, n] = Phase+p+js[itrc]  # +2*np.pi*kdotx[n]*FT[n]

    for itrc in range(0, wc.NC):
        AET_FTds[itrc, 0] = (AET_FTs[itrc, 1]-AET_FTs[itrc, 0]-FT[1]+FT[0])/wc.DT+FTd[0]
        AET_FTds[itrc, NT-1] = (AET_FTs[itrc, NT-1]-AET_FTs[itrc, NT-2]-FT[NT-1]+FT[NT-2])/wc.DT+FTd[NT-1]

    for n in range(1, NT-1):
        FT_shift = -FT[n+1]+FT[n-1]
        FTd_shift = FTd[n]
        for itrc in range(0, wc.NC):
            AET_FTds[itrc, n] = (AET_FTs[itrc, n+1]-AET_FTs[itrc, n-1]+FT_shift)/(2*wc.DT)+FTd_shift

@njit()
def TaylorF2_ref_time_guess(Mt,Mc,FI):
    """This is the TaylorF2 model to 2PN order. DOI: 10.1103/PhysRevD.80.084043"""
    eta = (Mc/Mt)**(5/3)

    if eta>=0.25:
        delta = 0.
    else:
        delta = np.sqrt(1-4*eta) #(m1-m2)/Mt


    c0 = 3/(128*eta)
    c1 = 20/9*(743/336+11/4*eta)


    p0 = 5*c0*Mt/6
    p1 = 3/5*c1

    nuI = (np.pi*Mt*FI)**(1/3)
    return p0/nuI**8*(1+p1*nuI**2)*nuI**6

def TaylorT3_ref_time_match(Mt,Mc,f_goal,t_guess):
    """This is the TaylorT3 model to 1PN order. DOI: 10.1103/PhysRevD.80.084043"""
    eta = (Mc/Mt)**(5/3)

    c0 = 1/(8*np.pi*Mt)
    c1 = (743/2688+11/32*eta)

    f_func = lambda th: f_goal-(c0*th**3*(1+c1*th**2))
    th_guess =  (eta*t_guess/(5*Mt))**(-1/8)
    th_ref = fsolve(f_func,th_guess)[0]
    TTRef = (5*Mt)/(eta*th_ref**8)
    return TTRef

@njit()
def AmpFreqDeriv_inplace(AS, PS, FS, FDS, FDDS,FDDDS, Amp, phi0, FI, FD0, FDD0,FDDD0, TTRef, TS, NT):
    """Get time domain waveform to lowest order, simple constant fdot"""
    # compute the intrinsic frequency, phase and amplitude
# , FDDD0
    phiRef = -phi0-2*np.pi*FI*(-TTRef)-np.pi*FD0*(-TTRef)**2-(np.pi/3)*FDD0*(-TTRef)**3-(np.pi/12)*FDDD0*(-TTRef)**4
    for n in range(0, NT):
        t = TS[n]
        FS[n] = FI+FD0*(t-TTRef) + (1/2)*FDD0*(t-TTRef)**2 + (1/6)*FDDD0*(t-TTRef)**3
        FDS[n] = FD0 + FDD0*(t-TTRef) + (1/2)*FDDD0*(t-TTRef)**2
        FDDS[n] = FDD0 + FDDD0*(t-TTRef)
        FDDDS[n] = FDDD0
        PS[n] = -phiRef-2*np.pi*FI*(t-TTRef)-np.pi*FD0*(t-TTRef)**2-(np.pi/3)*FDD0*(t-TTRef)**3 -(np.pi/12)*FDDD0*(t-TTRef)**4
        AS[n] = Amp

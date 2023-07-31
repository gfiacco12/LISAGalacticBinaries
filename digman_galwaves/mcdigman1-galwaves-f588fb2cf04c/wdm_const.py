"""C 2023 Matthew C. Digman
constants to speed up just in time compiler"""
import numpy as np

# global constants
SECSYEAR = (24*365*3600)  # Number of seconds in a calendar year #TODO check should be sidereal?
CLIGHT = 2.99792458e8     # Speed of light in m/s
AU = 1.4959787e11         # Astronomical Unit in meters
SQ3 = 1.732050807568877   # sqrt(3)


mode_select = 'GB'

if mode_select == 'GB':
    n_years = 4
    Nf = 2048         # frequency layers
    Nt = 512*n_years
    dt = 1.00250244140625*n_years*15./(2*Nf*Nt/2048**2)  # time cadence

    mult = 8          # over sampling

    Nsf = 150         # frequency steps
    Nfd = 10          # number of f-dots
    dfdot = 0.1       # fractional fdot increment
    Nfd_negative = 5
else:
    raise ValueError('unrecognized value for mode select')

nx = 4.0              # filter steepness in frequency


# LISA constants
Larm = 2.5e9          # Mean arm length of the LISA detector (meters)
Sps = 2.25e-22        # Photon shot noise power
Sacc = 9.0e-30        # Acceleration noise power
kappa0 = 0.0          # Initial azimuthal position of the guiding center
lambda0 = 0.0         # Initial orientation of the LISA constellation
fstr = 0.01908538064  # Transfer frequency
ec = 0.0048241852175  # LISA orbital eccentricity; should be Larm/(2*AU*np.sqrt(3))?
fm = 3.168753575e-8   # LISA modulation frequency


# derived constants
N = Nt*Nf             # total points
Tobs = dt*N           # duration
DT = dt*Nf            # width of wavelet pixel in time
DF = 1./(2*dt*Nf)     # width of wavelet pixel in frequency
K = mult*2*Nf         # filter length
Tw = dt*K             # filter duration
L = 512               # reduced filter length - must be a power of 2
p = K/L               # downsample factor
dom = 2.*np.pi/Tw     # angular frequency spacing
OM = np.pi/dt         # Nyquist angular frequency
DOM = OM/Nf           # 2 pi times DF
insDOM = 1./np.sqrt(DOM)
B = OM/(2*Nf)
A = (DOM-B)/2
BW = (A+B)/np.pi      # total width of wavelet in frequency
df = BW/Nsf

dfd = DF/Tw*dfdot

NC = 3                # number of TDI channels to use

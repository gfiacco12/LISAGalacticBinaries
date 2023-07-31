#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:13:30 2023

@author: noah
"""

import numpy as np

import matplotlib.pyplot as plt

import math

import WDMWaveletTransforms.wavelet_transforms as wt

# Define the wavelet transform parameters
Nt = 128                  # number of time pixels
Nf = 256                  # number of frequency pixels
dt = 60.                  # spacing of samples in time domain

ND = Nt*Nf                # total number of pixels
DT = dt*Nf                # width of time pixels
DF = 1./(2*dt*Nf)         # width of frequency pixels

SECSYEAR = (24*365*3600)  # Number of seconds in a calendar year

Tobs = dt*ND              # total observing time in seconds
TobsYr = Tobs/SECSYEAR    # total observing time in years
    
Fobs = Nf*DF # range of frequency (mHz)

f0 = 128 # central frequency pixel
t0 = 64 # central time pixel
q = 8 # time span parameter (?) only constricts < 12
q2 = (0.5*q*ND) # actual time span
N2 = (2*q2)/DT # float number of time pixels (-q,q)
N = int(N2/2)

print('Time pixel width (s): ' + str(DT))
print('Time span (-q,q): ' + str(2*q2+1))
print('Float number of pixels (-q,q): ' + str(N2))
print('Number of time pixels (-q,q): '+ str(2*N))

# define wavelet objects
wavelet = np.zeros((2*N+1,Nt,Nf))
wavelet_td = np.zeros((2*N+1,ND))
prod = np.zeros((2*N+1,ND))
cumsum = np.zeros((2*N+1,ND))

xx=np.arange(0,ND)*dt

#xx2 = 2*(xx-Tobs/2)/(q*ND)
xx2 = (xx-Tobs/2)/q2

# generate wavelets (-q,q)

for i in range(-N,N+1):
    wavelet[N+i][t0+i][f0]=1.
    print('Wavelet # '+str(i+N)+ ' centered: '+str(i)+'DT')
    i+=1

# transform wavelets to time domain

for i in range(0,2*N+1):
    wavelet_td[i][:] = wt.inverse_wavelet_time(wavelet[i][:][:],Nf,Nt,mult=q)
    print('wavelet # '+str(i) + ' transformed')
    #plt.plot(xx,wavelet_td[i][:])
    i+=1

# plot wavelets

for i in range(N,N+1):
    plt.plot((xx2),wavelet_td[i][:])
    plt.xlim(-1.5,1.5)
    plt.xlabel('Time (q)')
    plt.ylabel('Wavelet amplitude')

plt.show()

# taking the cumulative sum of the product for (-q,q)

for i in range(0,2*N+1):
    prod[i][:] = wavelet_td[N][:]*wavelet_td[i][:]
    cumsum[i][:] = np.cumsum(prod[i][:])
    cumsum[q][:]=0.
    plt.plot(xx2,cumsum[i][:])
    plt.xlabel('Time (q)')
    plt.ylabel('Cumulative Sum Amplitude')
    plt.title('Time shifts (-q,q), Frequency = ' + str(math.trunc(Fobs/2*1e3)) + 'mHz')
    plt.xlim(-1,1)
    i+=1
    
plt.show()

# taking the cumulative sum of the product for (-q,0)

for i in range(0,N):
    prod[i][:] = wavelet_td[N][:]*wavelet_td[i][:]
    cumsum[i][:] = np.cumsum(prod[i][:])
    cumsum[q][:]=0.
    plt.plot(xx2,cumsum[i][:],label='shift ' + str(-N+i) + 'DT')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xlabel('Time (q)')
    plt.ylabel('Cumulative Sum Amplitude')
    plt.title('Time shifts (-q,0), Frequency = ' + str(math.trunc(Fobs/2*1e3)) + 'mHz')
    plt.xlim(-1,1)
    i+=1
    
plt.show()

# taking the cumulative sum of the product for (0,q)

for i in range(q,2*N+1):
    prod[i][:] = wavelet_td[N][:]*wavelet_td[i][:]
    cumsum[i][:] = np.cumsum(prod[i][:])
    cumsum[q][:]=0.
    plt.plot(xx2,cumsum[i][:],label='shift ' + str(-N+i) + 'DT')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xlabel('Time (q)')
    plt.ylabel('Cumulative Sum Amplitude')
    plt.title('Time shifts (0,+q), Frequency = ' + str(math.trunc(Fobs/2*1e3)) + 'mHz')
    plt.xlim(-1,1)
    i+=1
    
plt.show()

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 12:40:08 2022

@author: arantt3
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

############# x parameters #############
# xmin = min value of x
xmin = -10.0
# xmax = max value of x
xmax = 10.0
# xpoints = number of points in x
xpoints = 400
# dx = step size in x
dx = (xmax - xmin) / xpoints
# Define x
x = np.linspace(xmin, xmax, xpoints+1)

############# t parameters #############
# tmin = min value of t
tmin = 0.0
# tmax = max value of t
tmax = 2.0
# tpoints = number of points in t
tpoints = 1000
# dt = step size in t
dt = (tmax - tmin) / tpoints

############# wave parameters ###########
# t = choose t value for time-dependent wavefunction
t = 0.0
# wvIndex = index for time-dependent wavefunction
wvIndex = int(t * (tpoints/tmax))

# Constants
p0 = 2.0
sigma = 0.5
analyticalConst = (1.0 + 1.0j*t) / (2.0*sigma)

######################## Defining Wavefunctions ##########################

# Define analytical time-dependent wavefunction and checking for its norm
Analytical = np.empty([xpoints+1], dtype = complex)
Analytical = ((np.sqrt(2.0*np.pi*sigma) * analyticalConst)**(-0.5)) * np.exp(-(x - p0*t)**2.0 / (4.0*(sigma)*analyticalConst)) * np.exp(1.0j*p0*x) * np.exp(-0.5j*(p0**2)*t)

norm = trapz(np.abs(Analytical)**2,x)
print('\nNorm of Analytical time-dependent wavefunction: ', norm)

# Defining wavefunction at t = 0 and checking its norm and expectation
psi0 = np.empty([xpoints+1], dtype = complex)
psi0 = ((1.0 / (2.0*np.pi*sigma)) ** (0.25)) * np.exp(1.0j*p0*x) * np.exp(-0.25*x**2 / sigma)

norm = trapz(np.abs(psi0)**2,x)
print('\nNorm of time-independent wavefunction: ', norm)
expect = trapz(x*np.abs(psi0**2),x)
print('\nExpectation value of time-independent wavefunction: ', expect)
print('\n')

#################### Solving Schrodinger's Equation ######################

# Define function for second derivative
def second_derivative(psi):
    seconddiv = np.empty([xpoints+1], dtype = complex)
    divconst = 1 / (dx**2.0)
    seconddiv[0] = divconst * (2.0*psi[0] - 5.0*psi[1] + 4.0*psi[2] - psi[3])
    for i in range(1, xpoints):
        seconddiv[i] = divconst * (psi[i-1] -2.0*psi[i] + psi[i+1])
    seconddiv[xpoints] = divconst * (2.0*psi[xpoints] - 5.0*psi[xpoints-1] + 4.0*psi[xpoints-2] - psi[xpoints-3])
    return seconddiv

# Applying Runge-Kutta to solve for time-dependent wavefunction
psinew = np.empty([xpoints+1], dtype = complex)
psinew = psi0

psi = []
psi.append(psinew)

for i in range(1, tpoints+1):
    c1 = np.empty([xpoints+1], dtype = complex)
    c2 = np.empty([xpoints+1], dtype = complex)
    c3 = np.empty([xpoints+1], dtype = complex)
    c4 = np.empty([xpoints+1], dtype = complex)
   
    c1 = 0.5j * second_derivative(psinew)
    c2 = 0.5j * second_derivative(psinew + 0.5*c1*dt)
    c3 = 0.5j * second_derivative(psinew + 0.5*c2*dt)
    c4 = 0.5j * second_derivative(psinew + c3*dt)
   
    psinew = psinew + (1.0/6.0)*dt*(c1 + 2.0*c2 + 2.0*c3 + c4)
    psi.append(psinew)

# Normalizing wavefunction for all times
for i in range(0, tpoints+1):
    norm = trapz(np.abs(psi[i])**2,x)
    psi[i] = (1.0/norm) * psi[i]
   
######################## Checking Errors ##########################

# Checking norm for wavefunction at different times
print('\nNorm of wavefunction at different t values (from 0 to 2 in increments of 0.05):\n')
for i in range(0, tpoints+1, 50):
    norm = trapz(np.abs(psi[i])**2,x)
    print(norm)
print('\n')

# Calculating error for time-dependent wavefunction
errorreal = np.empty([xpoints+1])
errorimag = np.empty([xpoints+1])
errorprob = np.empty([xpoints+1])
Q = np.abs(psi[wvIndex])**2
Qprime = np.abs(Analytical)**2
for i in range (0,xpoints+1):
    errorreal[i] = abs(psi[wvIndex].real[i] - Analytical.real[i])
    errorimag[i] = abs(psi[wvIndex].imag[i] - Analytical.imag[i])
    errorprob[i] = abs(Q[i] - Qprime[i])
   
############################## Plots ################################
   

# Plot of Real part of Wavefunction
plt.plot(x, psi[wvIndex].real, label = "Calculated")
plt.plot(x, Analytical.real, '--', label = "Analytical")
plt.xlabel('x')
plt.ylabel('Re Psi(x, t))')
plt.title('Plot of Real part of wavefunction at t = '+str(t))
plt.legend()
plt.show()

# Plot of error in Real part of Wavefunction
plt.plot(x, errorreal)
plt.xlabel('x')
plt.ylabel('error')
plt.title('Plot of error of Real part of wavefunction at t = '+str(t))
plt.show()

# Plot of Imaginary part of Wavefunction
plt.plot(x, psi[wvIndex].imag, label = "Calculated")
plt.plot(x, Analytical.imag, '--', label = "Analytical")
plt.xlabel('x')
plt.ylabel('Im Psi(x, t))')
plt.title('Plot of Imaginary part of wavefunction at t = '+str(t))
plt.legend()
plt.show()

# Plot of error in Imaginary part of Wavefunction
plt.plot(x, errorimag)
plt.xlabel('x')
plt.ylabel('error')
plt.title('Plot of error of Imaginary part of wavefunction at t = '+str(t))
plt.show()

#  Plot of Probability Density
plt.plot(x, np.conj(psi[wvIndex])*psi[wvIndex], label = "Calculated")
plt.plot(x, np.conj(Analytical)*Analytical, '--', label = "Analytical")
plt.xlabel('x')
plt.ylabel('Probability')
plt.title('Plot of Probability Density at t = '+str(t))
plt.legend()
plt.show()

# Plot of error in Probability Density
plt.plot(x, errorprob)
plt.xlabel('x')
plt.ylabel('error')
plt.title('Plot of error of Probability Density at t = '+str(t))
plt.show()
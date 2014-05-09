#!/usr/bin/env python

import scipy.stats
import numpy as np
import astropy.io.fits as fits
import emcee
import matplotlib.pyplot as p
from matplotlib import rc
from astropy.table import Table, Column
rc('text',usetex=True)



def logprob3d(p,x,y,z,x_err,y_err,z_err):
    theta,phi,scatter = p[0],p[1],p[2]
    if np.abs(theta-np.pi/2)>np.pi/2:
        return -np.inf
    if np.abs(phi-np.pi/2)>np.pi/2:
        return -np.inf
    if scatter<0.0:
        return -np.inf
# Distance between ray at theta, phi and a point x,y,z
#Gamma is the dot product of the data vector along the theoretical line
    Gamma = x*np.sin(theta)*np.cos(phi)+\
        y*np.sin(theta)*np.sin(phi)+z*np.cos(theta)
    DeltaX2 = (x-Gamma*np.sin(theta)*np.cos(phi))**2
    DeltaY2 = (y-Gamma*np.sin(theta)*np.sin(phi))**2
    DeltaZ2 = (z-Gamma*np.cos(theta))**2
    Delta2 = DeltaX2+DeltaY2+DeltaZ2
    Sigma = DeltaX2/Delta2*x_err**2+\
        DeltaY2/Delta2*y_err**2+\
        DeltaZ2/Delta2*z_err**2+scatter
#    print(Sigma)
    lp = -0.5*np.nansum(Delta2/Sigma)-np.nansum(np.log(Sigma))

    return lp

def logprob3d_xoff(p,x,y,z,x_err,y_err,z_err):
    theta,phi,scatter,xoffset = p[0],p[1],p[2],p[3]
    if np.abs(theta-np.pi/4)>np.pi/4:
        return -np.inf
    if np.abs(phi-np.pi/4)>np.pi/4:
        return -np.inf
    if scatter<0.0:
        return -np.inf
# Distance between ray at theta, phi and a point x,y,z
#Gamma is the dot product of the data vector along the theoretical line
    Gamma = (x+xoffset)*np.sin(theta)*np.cos(phi)+\
        y*np.sin(theta)*np.sin(phi)+z*np.cos(theta)
    DeltaX2 = (x+xoffset-Gamma*np.sin(theta)*np.cos(phi))**2
    DeltaY2 = (y-Gamma*np.sin(theta)*np.sin(phi))**2
    DeltaZ2 = (z-Gamma*np.cos(theta))**2
    Delta2 = DeltaX2+DeltaY2+DeltaZ2
    Sigma = DeltaX2/Delta2*x_err**2+\
        DeltaY2/Delta2*y_err**2+\
        DeltaZ2/Delta2*z_err**2+scatter
    lp = -0.5*np.nansum(Delta2/Sigma)-0.5*np.log(scatter)*(Delta2.size)
#    print(Sigma)
    return lp


def logprob2d(p,x,y,x_err,y_err):
    theta,scatter = p[0],p[1]
    if np.abs(theta-np.pi/4)>np.pi/4:
        return -np.inf
    if scatter<0.0:
        return -np.inf
    #    if width<0.0:
    #        return -np.inf

    scatter = 0.0
    displacement = 0.0
    # Displacement away from the line
    Delta = (np.cos(theta)*y - np.sin(theta)*x)**2
    # Displacement from zero
    #    delta = (np.sin(theta)*y + np.cos(theta)*x-displacement)**2
    Sigma = (np.sin(theta))**2*x_err**2+(np.cos(theta))**2*y_err**2+scatter
    #    sigma = (np.cos(theta))**2*x_err**2+(np.sin(theta))**2*y_err**2+width

    term1 = -0.5*np.nansum(Delta/Sigma)
    #    term2 = -np.log(scatter*len(x))*0.5
    term2 = 0
    #    term2 = -np.nansum(np.log((np.sin(theta))**2*x_err**2 +\
    #                       (np.cos(theta))**2*y_err**2+scatter))*0.5


    #   if term2 > 0.5*term1:
    #       term2 = term1

    #    lp = -0.5*np.nansum(Delta/Sigma)-np.nansum(np.log(x_err**2 + y_err**2+scatter))*0.5#-\
    lp = term1 + term2
    
    #        0.5*np.nansum(delta/sigma)-np.nansum(np.log(sigma))
    #    lp =  -0.5*np.nansum(Delta/Sigma)-0.5*np.log(scatter)*Sigma.size
#    lp = -0.5*np.nansum(Delta/Sigma)-0.5*scipy.stats.norm.logpdf(scatter,0,1.0)
    return lp

def logprob2dcc(p,x,y,x_err,y_err):
    theta,scatter,rho = p[0],p[1],p[2]
    if np.abs(theta-np.pi/4)>np.pi/4:
        return -np.inf
    if scatter<0.0:
        return -np.inf
    if abs(rho)>1.0:
        return -np.inf

#    if width<0.0:
#        return -np.inf
    sigxy = rho*x_err*y_err
    displacement = 0.0
# Displacement away from the line
    Delta = (np.cos(theta)*y - np.sin(theta)*x)**2
# Displacement from zero
#    delta = (np.sin(theta)*y + np.cos(theta)*x-displacement)**2
    Sigma = (np.sin(theta))**2*x_err**2+(np.cos(theta))**2*y_err**2+\
        scatter-2*np.sin(theta)*np.cos(theta)*sigxy
#    sigma = (np.cos(theta))**2*x_err**2+(np.sin(theta))**2*y_err**2+width

    lp = -0.5*np.nansum(Delta/Sigma)-np.nansum(np.log(Sigma))*0.5#-\
#        0.5*np.nansum(delta/sigma)-np.nansum(np.log(sigma))
#    lp =  -0.5*np.nansum(Delta/Sigma)-0.5*np.log(scatter)*Sigma.size
#    lp = -0.5*np.nansum(Delta/Sigma)-0.5*scipy.stats.norm.logpdf(scatter,0,1.0)
    return lp


def logprob2dns(p,x,y,x_err,y_err):
    theta = p[0]
    if np.abs(theta-np.pi/4)>np.pi/4:
        return -np.inf

# Displacement away from the line
    Delta = (np.cos(theta)*y - np.sin(theta)*x)**2
# Displacement from zero
#    delta = (np.sin(theta)*y + np.cos(theta)*x)**2
    Sigma = (np.sin(theta))**2*x_err**2+(np.cos(theta))**2*y_err**2
#    sigma = (np.cos(theta))**2*x_err**2+(np.sin(theta))**2*y_err**2+width
    lp = -0.5*np.nansum(Delta/Sigma)-np.nansum(np.log(Sigma))
#    lp =  -0.5*np.nansum(Delta/Sigma)-0.5*np.log(scatter)*Sigma.size
#    lp = -0.5*np.nansum(Delta/Sigma)-0.5*scipy.stats.norm.logpdf(scatter,0,1.0)
    return lp

def logprob2dslope(p,x,y,x_err,y_err):
    m,scatter = p[0],p[1]
    if scatter<0.0:
        return -np.inf
    sigma = (scatter+y_err**2+m**2*x_err**2)
    lp = -0.5*np.nansum((y-m*x)**2/sigma)-0.5*np.nansum(np.log(sigma))
    return lp


def logprob2d_xoff(p,x,y,x_err,y_err):
    theta,scatter,xoff = p[0],p[1],p[2]
    if np.abs(theta-np.pi/4)>np.pi/4:
        return -np.inf
    if scatter<0.0:
        return -np.inf
    Delta = (np.cos(theta)*y - np.sin(theta)*(x+xoff))**2
    Sigma = (np.sin(theta))**2*x_err**2+(np.cos(theta))**2*y_err**2+scatter
    lp = -0.5*np.nansum(Delta/Sigma)-0.5*np.nansum(np.log(Sigma))
    return lp


# x = np.random.randn(100)*1
# y = np.random.randn(100)*1
# z = np.random.randn(100)*1
# x_err = np.ones(100)*1
# y_err = np.ones(100)*2
# z_err = np.ones(100)*1


# #p0 = np.array([np.pi/4,np.pi/4,0,0])
# #lp = logprob3d_xoff(p0,x,y,z,x_err,y_err,z_err)

# #print(lp)


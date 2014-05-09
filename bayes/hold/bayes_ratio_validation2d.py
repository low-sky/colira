#!/usr/bin/env python

import scipy.stats
import numpy as np
import astropy.io.fits as fits
import emcee
import matplotlib.pyplot as p
from matplotlib import rc
from astropy.table import Table, Column
rc('text',usetex=True)

execfile('logprob.py')

s = fits.getdata('colira_subset.fits')
hdr = fits.getheader('colira_subset.fits')
GalNames = np.unique(s['GALNAME'])

nTest = 20
nPoints = 100
Signal = np.linspace(1,20,nTest)
t = Table(names=('Name','theta','theta+','theta-','phi','phi+','phi-',\
                     'R21-','R21','R21+','R32-','R32','R32+','sigx',\
                     'R31-','R31','R31+','sigy'),\
              dtypes=('S7','f8','f8','f8','f8','f8','f8','f8','f8','f8',\
                          'f8','f8','f8','f8','f8','f8','f8','f8'))
r21True = 0.7
r32True = 0.5
scatter =2
badfrac = 0.15
BadMult = 5

for ctr in np.arange(nTest):
    x = np.abs((np.random.randn(nPoints)))*Signal[ctr]
    y = r21True*x
    z = r21True*r32True*x
# Add noise
    x = x+np.random.randn(nPoints)*5+np.random.randn(nPoints)*scatter/np.sqrt(2)
    y = y+np.random.randn(nPoints)+np.random.randn(nPoints)*scatter/np.sqrt(2)
    z = z+np.random.randn(nPoints)
    nBad = len(x)*badfrac
    
    x_err = np.ones((nPoints))*5
    y_err = np.ones((nPoints))
    z_err = np.ones((nPoints))

    x[0:nBad] = x[0:nBad]+np.random.randn(nBad)*BadMult*x_err[0:nBad]
    y[0:nBad] = y[0:nBad]+np.random.randn(nBad)*BadMult*y_err[0:nBad]
    z[0:nBad] = z[0:nBad]+np.random.randn(nBad)*BadMult*z_err[0:nBad]
    
    ndim, nwalkers = 7,50

    p0 = np.zeros((nwalkers,ndim))
    p0[:,0] = np.random.rand(nwalkers)*np.pi/2
    p0[:,1] = (np.random.randn(nwalkers))**2
    p0[:,2] = (np.random.randn(nwalkers))**2
    p0[:,3] = (np.random.randn(nwalkers))
    p0[:,4] = (np.random.randn(nwalkers))**2*0.05
    p0[:,5] = (np.random.randn(nwalkers))
    p0[:,6] =(np.random.randn(nwalkers))**2*10
    sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                    logprob2d_xoff_scatter_mixture,
                                    args=[x,y,x_err,y_err])
    pos, prob, state = sampler.run_mcmc(p0, 400)
    sampler.reset()
    sampler.run_mcmc(pos, 1000)

    print(ctr,np.mean(sampler.acceptance_fraction),np.tan(np.median(sampler.flatchain[:,0])))
    t.add_row()
    t['theta'][ctr] = np.median(sampler.flatchain[:,0])
    t['theta+'][ctr] = scipy.stats.scoreatpercentile(\
        sampler.flatchain[:,0],85) - t['theta'][ctr]
    t['theta-'][ctr] = t['theta'][ctr] - \
        scipy.stats.scoreatpercentile(\
        sampler.flatchain[:,0],15)
    
    
    r32 = np.tan(sampler.flatchain[:,0])
    t['R32'][ctr] = np.median(r32)
    t['R32+'][ctr] = scipy.stats.scoreatpercentile(r32,85)
    t['R32-'][ctr] = scipy.stats.scoreatpercentile(r32,15)
    
    # r31 = 1/np.tan(sampler.flatchain[:,0])/np.cos(sampler.flatchain[:,1])
    # t['R31'][ctr] = np.median(r31)
    # t['R31+'][ctr] = scipy.stats.scoreatpercentile(r31,85)
    # t['R31-'][ctr] = t['R31'][ctr] =scipy.stats.scoreatpercentile(r31,1)
    # t['var'][ctr] = np.median(sampler.flatchain[:,2])

p.figure(1)
# p.subplot(121)    
# p.errorbar(r21True,t['R21'],yerr=[t['R21']-t['R21-'],t['R21+']-t['R21']],linestyle='none')    
# p.plot(r21True,r21True)
# p.xlabel(r'$R_{21}$ True')
# p.ylabel(r'$R_{21}$ Estimate')

p.subplot(111)    
p.errorbar(Signal,t['R32'],yerr=[t['R32']-t['R32-'],t['R32+']-t['R32']],linestyle='none')    

p.xlabel(r'$R_{32}$ True')
p.ylabel(r'$R_{32}$ Estimate')

p.show()


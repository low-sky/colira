#!/usr/bin/env python

import numpy as np
import astropy.io.fits as fits
import emcee
import matplotlib.pyplot as p

def logprob(p,x,y,x_err,y_err):
    theta,scatter = p[0],p[1]
    if np.abs(theta-np.pi/4)>np.pi/4:
        return -np.inf
    Delta = (np.cos(theta)*y - np.sin(theta)*x)**2
    Sigma = (np.sin(theta))**2*x_err**2+(np.cos(theta))**2*y_err**2+scatter**2
    lp = -0.5*np.nansum(Delta/Sigma)-0.5*np.nansum(np.log(Sigma))

    return lp

s = fits.getdata('colira_subset.fits')
hdr = fits.getheader('colira_subset.fits')
GalNames = np.unique(s['GALNAME'])
figdir = '../plots/'
for name in GalNames: 

    idx = np.where(s['GALNAME']==name)
    sub = s[idx]

    x = sub['CO21']
    x_err = sub['CO21_ERR']
    y = sub['CO32']
    y_err = sub['CO32_ERR']

    ndim, nwalkers = 2,100
    p0 = np.zeros((nwalkers,2))
    p0[:,0] = np.pi/4+np.random.randn(nwalkers)*0.1
    p0[:,1] = 1.0 + np.random.randn(nwalkers)*0.2
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, 
                                    args=[x,y,x_err,y_err])
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(pos, 1000)

    p.figure(1)
#    p.subplot(321)
#    p.plot(np.tan(sampler.flatchain[:,0]))
#    p.subplot(322)
#    p.plot(np.abs(sampler.flatchain[:,1]))

    p.subplot(323)
    p.hist(np.tan(sampler.flatchain[:,0]))
    p.subplot(324)
    p.hist(np.abs(sampler.flatchain[:,1]))
    p.subplot(325)
    p.errorbar(x,y,xerr=x_err,yerr=y_err,fmt=None,marker=None,mew=0)
    p.scatter(x,y)
    testx = np.linspace(np.nanmin(x),np.nanmax(x),10)
    p.plot(testx,np.tan(np.median(sampler.flatchain[:,0]))*testx,color='r')
    p.savefig(figdir+name+'.pdf',format='pdf',orientation='portrait')
    p.close()
    p.clf()
    print(name,np.mean(sampler.acceptance_fraction))

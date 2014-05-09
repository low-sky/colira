#!/usr/bin/env python

import numpy as np
import scipy.stats
import astropy.io.fits as fits
import emcee
import matplotlib.pyplot as p
from matplotlib import rc

execfile('logprob.py')

s = fits.getdata('colira_subset.fits')
hdr = fits.getheader('colira_subset.fits')
GalNames = np.unique(s['GALNAME'])

idx = np.where(np.isfinite(s['CO10']) & np.isfinite(s['CO21'])\
                   & np.isfinite(s['CO32']))
sub = s[idx]
if len(sub)>1:
    x = sub['CO10']
    x_err = sub['CO10_ERR']
    
    y = sub['CO21']
    y_err = sub['CO21_ERR']
    z = sub['CO32']
    z_err = sub['CO32_ERR']

    ndim, nwalkers = 3,50
    p0 = np.zeros((nwalkers,3))
    p0[:,0] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
    p0[:,1] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
    p0[:,2] = (np.random.random(nwalkers))*20
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob3d, 
                                    args=[x,y,z,x_err,y_err,z_err],threads=12)
    pos, prob, state = sampler.run_mcmc(p0, 200)
    sampler.reset()
    sampler.run_mcmc(pos, 1000)


    # p.figure(1)
    # p.subplot(321)
    # p.plot(1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1]))
    # p.subplot(322)
    # p.plot(np.tan(sampler.flatchain[:,1]))
    # p.subplot(323)
    # p.hist(1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1]))
    # p.subplot(324)
    # p.hist(np.tan(sampler.flatchain[:,1]))
    # p.subplot(325)
    # p.errorbar(x,y,xerr=x_err,yerr=y_err,fmt=None,marker=None,mew=0)
    # p.scatter(x,y,marker='.')
    # testx = np.linspace(np.nanmin(x),np.nanmax(x),10)
    # p.plot(testx,np.tan(np.median(sampler.flatchain[:,1]))*testx,color='r')
    # p.subplot(326)
    # p.hexbin(1/np.tan(sampler.flatchain[:,0])/\
    #              np.sin(sampler.flatchain[:,1]),\
    #              np.tan(sampler.flatchain[:,1]))

    hist,xe,ye = np.histogram2d(np.tan(sampler.flatchain[:,1]),\
                                    1/np.tan(sampler.flatchain[:,0])/\
                                    np.sin(sampler.flatchain[:,1]),\
                                    bins=200,range=[[0.6,0.8],[0.5,0.7]])
    p.contour(xe[1:],ye[1:],hist,colors='black')
    p.xlabel(r'$R_{21}$')
    p.ylabel(r'$R_{32}$')
    p.savefig('bayes_ratio_all.pdf',format='pdf',orientation='portrait')
    p.close()

#    print(name,np.mean(sampler.acceptance_fraction),np.tan(np.median(sampler.flatchain[:,1])))

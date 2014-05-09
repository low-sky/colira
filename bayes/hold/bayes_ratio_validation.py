#!/usr/bin/env python

import scipy.stats
import numpy as np
import astropy.io.fits as fits
import emcee
import matplotlib.pyplot as p
from matplotlib import rc
from astropy.table import Table, Column
rc('text',usetex=True)

#execfile('logprob.py')
execfile('lptest.py')
s = fits.getdata('colira_subset.fits')
hdr = fits.getheader('colira_subset.fits')
GalNames = np.unique(s['GALNAME'])

nTest = 10
nPoints = 100
Signal = 1
t = Table(names=('Name','theta','theta+','theta-','phi','phi+','phi-',\
                     'R21-','R21','R21+','R32-','R32','R32+','var',\
                     'R31-','R31','R31+'),\
              dtypes=('S7','f8','f8','f8','f8','f8','f8','f8','f8','f8',\
                          'f8','f8','f8','f8','f8','f8','f8'))
r21True = np.random.random(nTest)
r32True = np.random.random(nTest)
for ctr in np.arange(nTest):
    x = np.abs(np.random.randn(nPoints))*Signal
    y = r21True[ctr]*x
    z = r21True[ctr]*r32True[ctr]*x
# Add noise
    x = x+np.random.randn(nPoints)*5
    y = y+np.random.randn(nPoints)
    z = z+np.random.randn(nPoints)

    x_err = np.ones((nPoints))*5
    y_err = np.ones((nPoints))
    z_err = np.ones((nPoints))

    ndim, nwalkers = 3,50
    p0 = np.zeros((nwalkers,3))
    p0[:,0] = np.random.random(nwalkers)*np.pi/2
    p0[:,1] = np.random.random(nwalkers)*np.pi/2

#    p0[:,1] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
#    p0[:,2] = (np.random.random(nwalkers))*20
#    p0[:,3] = (np.random.randn(nwalkers))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob3d, 
                                 args=[x,y,z,x_err,y_err,z_err])
    pos, prob, state = sampler.run_mcmc(p0, 200)
    sampler.reset()
    sampler.run_mcmc(pos, 1000)
    p.figure(1)
    p.subplot(241)
    
    p.plot(sampler.flatchain[:,0])
    p.ylabel(r'$\theta$')
    
    p.subplot(242)
    
    p.plot(sampler.flatchain[:,1])
    p.ylabel(r'$\phi$')
    
    p.subplot(243)
    p.hist(1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1]),\
               range=[0,2],bins=100)
    p.xlabel('$R_{32}$')
    p.subplot(244)
    p.hist(np.tan(sampler.flatchain[:,1]),range=[0,2],bins=100)
    p.xlabel('$R_{21}$')
    p.subplot(245)
    p.errorbar(x,y,xerr=x_err,yerr=y_err,fmt=None,marker=None,mew=0)
    p.scatter(x,y,marker='.')
    p.xlabel('CO(1-0)')
    p.ylabel('CO(2-1)')
    testx = np.linspace(np.nanmin(x),np.nanmax(x),10)
#    xoff = np.median(sampler.flatchain[:,3])
    xoff = 0
    p.plot(testx,np.tan(np.median(sampler.flatchain[:,1]))*(testx+xoff),color='r')
    
    p.subplot(246)
    p.errorbar(y,z,xerr=y_err,yerr=y_err,fmt=None,marker=None,mew=0)
    p.scatter(y,z,marker='.')
    p.xlabel('CO(2-1)')
    p.ylabel('CO(3-2)')
    testx = np.linspace(np.nanmin(y),np.nanmax(y),10)
    p.plot(testx,testx/np.tan(np.median(sampler.flatchain[:,0]))/\
               np.sin(np.median(sampler.flatchain[:,1])),color='r')
    
    p.subplot(247)
    p.plot(sampler.flatchain[:,2])
    p.xlabel(r'$V$')
    
    p.subplot(248)
    p.hexbin(np.tan(sampler.flatchain[:,1]),sampler.flatchain[:,2])
    p.xlabel('$R_{21}$')
    p.ylabel('Offset')
    p.savefig('validation_'+np.array_str(ctr)+'.pdf',format='pdf',orientation='portrait')
    
    p.close()
    
    print(ctr,np.mean(sampler.acceptance_fraction),1/np.tan(np.median(sampler.flatchain[:,0])))
    t.add_row()
    t['theta'][ctr] = np.median(sampler.flatchain[:,0])
    t['theta+'][ctr] = scipy.stats.scoreatpercentile(\
        sampler.flatchain[:,0],85) - t['theta'][ctr]
    t['theta-'][ctr] = t['theta'][ctr] - \
        scipy.stats.scoreatpercentile(\
        sampler.flatchain[:,0],15)
    t['phi'][ctr] = np.median(sampler.flatchain[:,1])
    t['phi+'][ctr] = scipy.stats.scoreatpercentile(\
        sampler.flatchain[:,1],85) - t['phi'][ctr]
    t['phi-'][ctr] = t['phi'][ctr] - \
        scipy.stats.scoreatpercentile(\
        sampler.flatchain[:,1],15)
    
    r21 = np.tan(sampler.flatchain[:,1])
    t['R21'][ctr] = np.median(r21)
    t['R21+'][ctr] = scipy.stats.scoreatpercentile(r21,85)
    t['R21-'][ctr] = scipy.stats.scoreatpercentile(r21,15)
    
    
    r32 = 1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1])
    t['R32'][ctr] = np.median(r32)
    t['R32+'][ctr] = scipy.stats.scoreatpercentile(r32,85)
    t['R32-'][ctr] = scipy.stats.scoreatpercentile(r32,15)
    
    r31 = 1/np.tan(sampler.flatchain[:,0])/np.cos(sampler.flatchain[:,1])
    t['R31'][ctr] = np.median(r31)
    t['R31+'][ctr] = scipy.stats.scoreatpercentile(r31,85)
    t['R31-'][ctr] = t['R31'][ctr] =scipy.stats.scoreatpercentile(r31,15)
    t['var'][ctr] = np.median(sampler.flatchain[:,2])

p.figure(1)
p.subplot(121)    
p.errorbar(r21True,t['R21'],yerr=[t['R21']-t['R21-'],t['R21+']-t['R21']],linestyle='none')    
p.plot(r21True,r21True)
p.xlabel(r'$R_{21}$ True')
p.ylabel(r'$R_{21}$ Estimate')

p.subplot(122)    
p.errorbar(r32True,t['R32'],yerr=[t['R32']-t['R32-'],t['R32+']-t['R32']],linestyle='none')    
p.plot(r32True,r32True)
p.xlabel(r'$R_{32}$ True')
p.ylabel(r'$R_{32}$ Estimate')

p.show()


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

s = fits.getdata('colira_subset12.fits')
hdr = fits.getheader('colira_subset12.fits')
GalNames = np.unique(s['GALNAME'])


t = Table(names=('Name','phi','phi+','phi-',\
                     'R21-','R21','R21+','var','xoff'),\
              dtypes=('S7','f8','f8','f8','f8','f8','f8','f8','f8'))
for tag in t.keys():
    if tag != 'Name':
        t[tag].format = '{:.3f}'

nGal = len(GalNames)

it = np.nditer(GalNames,flags=['f_index'])

while not it.finished:
    ctr = it.index
    name = np.array_str(it.value)
    t.add_row()
    t['Name'][ctr] = name.upper()
    idx = np.where(s['GALNAME']==name)
    sub = s[idx]
    if len(sub)>1:
        x = sub['CO10']
        x_err = sub['CO10_ERR']
    
        y = sub['CO21']
        y_err = sub['CO21_ERR']

        ndim, nwalkers = 3,50
        p0 = np.zeros((nwalkers,ndim))
        p0[:,0] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
        p0[:,1] = (np.random.random(nwalkers))*20
        p0[:,1] = (np.random.randn(nwalkers))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob2d_xoff, 
                                    args=[x,y,x_err,y_err])
        pos, prob, state = sampler.run_mcmc(p0, 200)
        sampler.reset()
        sampler.run_mcmc(pos, 1000)
        print(name,np.mean(sampler.acceptance_fraction),np.tan(np.median(sampler.flatchain[:,0])))

        t['phi'][ctr] = np.median(sampler.flatchain[:,0])
        t['phi+'][ctr] = scipy.stats.scoreatpercentile(\
            sampler.flatchain[:,0],85) - t['phi'][ctr]
        t['phi-'][ctr] = t['phi'][ctr] - \
            scipy.stats.scoreatpercentile(\
            sampler.flatchain[:,0],15)

        r21 = np.tan(sampler.flatchain[:,0])
        t['R21'][ctr] = np.median(r21)
        t['R21+'][ctr] = scipy.stats.scoreatpercentile(r21,85)
        t['R21-'][ctr] = scipy.stats.scoreatpercentile(r21,15)
        t['var'][ctr] = np.median(sampler.flatchain[:,1])
        t['xoff'][ctr] = np.median(sampler.flatchain[:,2])
    it.iternext()

t2 = Table(t)
t2.remove_columns(('phi','phi+','phi-','var'))
emptystring = np.empty((len(t2)),dtype='string')
emptystring[:]=''
t2.write('ratios21.tex',format='latex')

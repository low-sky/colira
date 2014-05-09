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

s = fits.getdata('colira.fits')
hdr = fits.getheader('colira.fits')
GalNames = np.unique(s['GALNAME'])
figdir ='../plots/'
cut = 1
t = Table(names=('Name','theta','theta+','theta-','phi','phi+','phi-',\
                     'R21-','R21','R21+','R32-','R32','R32+',\
                     'R31-','R31','R31+','var','xoff'),\
              dtypes=('S7','f8','f8','f8','f8','f8','f8','f8','f8','f8',\
                          'f8','f8','f8','f8','f8','f8','f8','f8'))
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
    idx = np.where((s['GALNAME']==name)*np.logical_and(s['CO21']>cut*s['CO21_ERR'],(s['CO32']>cut*s['CO32_ERR'])))
    sub = s[idx]
    if len(sub)>1:
        x = sub['CO21']
        x_err = sub['CO21_ERR']
        y = sub['CO32']
        y_err = sub['CO32_ERR']
        ndim, nwalkers = 2,50
        p0 = np.zeros((nwalkers,ndim))
        p0[:,0] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
        p0[:,1] = np.abs((np.random.randn(nwalkers)))*2
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob2d_scatter, 
                                    args=[x,y,x_err,y_err])
        pos, prob, state = sampler.run_mcmc(p0, 200)
        sampler.reset()
        sampler.run_mcmc(pos, 1000)

#######################
# PLOTTING
#######################
        p.figure(figsize=(6,6))

        p.subplot(221)
        p.hist(np.tan(sampler.flatchain[:,0]),range=[0,2],bins=100)
        p.xlabel('$R_{32}$')

        p.subplot(222)
        p.errorbar(x,y,xerr=x_err,yerr=y_err,fmt=None,marker=None,mew=0)
        p.scatter(x,y,marker='.')
        p.xlabel('CO(2-1)')
        p.ylabel('CO(3-2)')
        testx = np.linspace(np.nanmin(x),np.nanmax(x),10)
        xoff = 0
        p.plot(testx,np.tan(np.median(sampler.flatchain[:,0]))*(testx+xoff),color='r')

        p.subplot(223)
        p.hexbin(np.tan(sampler.flatchain[:,0]),np.sqrt(sampler.flatchain[:,1]))
        p.xlabel('$R_{32}$')
        p.ylabel('Scatter')

        p.tight_layout()
        p.savefig(figdir+name+'.32.xoff.pdf',format='pdf',orientation='portrait')

        p.close()
        p.clf()
        print(name,np.mean(sampler.acceptance_fraction),np.tan(np.median(sampler.flatchain[:,0])))

        t['phi'][ctr] = np.median(sampler.flatchain[:,0])
        t['phi+'][ctr] = scipy.stats.scoreatpercentile(\
            sampler.flatchain[:,0],85) - t['phi'][ctr]
        t['phi-'][ctr] = t['phi'][ctr] - \
            scipy.stats.scoreatpercentile(\
            sampler.flatchain[:,0],15)

        r32 = np.tan(sampler.flatchain[:,0])
        t['R32'][ctr] = np.median(r32)
        t['R32+'][ctr] = scipy.stats.scoreatpercentile(r32,85)
        t['R32-'][ctr] = scipy.stats.scoreatpercentile(r32,15)

        t['var'][ctr] = np.median(sampler.flatchain[:,1])
    it.iternext()

t['var'] = np.sqrt(t['var'])
t.write('br32.txt',format='ascii')
t2 = Table(t)
t2.remove_columns(('theta','phi','theta+','phi+','theta-','phi-'))
emptystring = np.empty((len(t2)),dtype='string')
emptystring[:]=''
col = Column(name='blank',data=emptystring)
t2.add_column(col,index=4)
col = Column(name='blank2',data=emptystring)
t2.add_column(col,index=8)
t2.write('r32_xoff_scatter.tex',format='latex')

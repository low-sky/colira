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
execfile('sampler_plot.py')
s = fits.getdata('colira.fits')
hdr = fits.getheader('colira.fits')
GalNames = np.unique(s['GALNAME'])

cut = -10
t = Table(names=('Name','theta','theta+','theta-','phi','phi+','phi-',\
                     'xR21-','xR21','xR21+','xR32-','xR32','xR32+',
                     'var32','var21',
                     'R21-','R21','R21+','R32-','R32','R32+',
                     'R31-','R31','R31+','var','xoff','Npts'),\
              dtypes=('S7','f8','f8','f8','f8','f8','f8','f8','f8','f8',
                      'f8','f8','f8','f8','f8','f8','f8','f8',
                      'f8','f8','f8','f8','f8','f8','f8','f8','f8'))
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
    idx = np.where((s['GALNAME']==name)*
                   np.logical_and(
            np.logical_and((s['CO10']>cut*s['CO10_ERR']), 
                           (s['CO21']>cut*s['CO21_ERR'])),
            (s['CO32']>cut*s['CO32_ERR'])))
    sub = s[idx]

    idx21 = np.where((s['GALNAME']==name)*
                (s['CO10']>cut*s['CO10_ERR'])*(s['CO21']>cut*s['CO21_ERR']))
    sub21 = s[idx21]

    idx32 = np.where((s['GALNAME']==name)*
                    (s['CO32']>cut*s['CO32_ERR'])*(s['CO21']>cut*s['CO21_ERR']))
        
    sub32 = s[idx32]
    print(len(sub21),len(sub32),len(sub))
    if len(sub)>1:
        x = sub['CO10']
        x_err = sub['CO10_ERR']
        y = sub['CO21']
        y_err = sub['CO21_ERR']
        z = sub['CO32']
        z_err = sub['CO32_ERR']
        ndim, nwalkers = 5,50
        p0 = np.zeros((nwalkers,ndim))
        p0[:,0] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
        p0[:,1] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
        p0[:,2] = (np.random.randn(nwalkers))**2
        p0[:,3] = (np.random.randn(nwalkers))**2
        p0[:,4] = (np.random.randn(nwalkers))**2
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob3d, 
                                    args=[x,y,z,x_err,y_err,z_err])
        pos, prob, state = sampler.run_mcmc(p0, 400)
        sampler.reset()
        sampler.run_mcmc(pos,1000)
        print(name,np.mean(sampler.acceptance_fraction),
              1/np.tan(np.median(sampler.flatchain[:,0])))

        sampler_plot(sampler,name=name)

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
            scipy.stats.scoreatpercentile(
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
        t['R31-'][ctr] = scipy.stats.scoreatpercentile(r31,15)
        t['var'][ctr] = np.median(sampler.flatchain[:,2]**2+
                sampler.flatchain[:,3]**2+
                sampler.flatchain[:,4]**2)
        t['Npts'][ctr] = len(x)


    if len(sub21)>1:
        x = sub21['CO10']
        x_err = sub21['CO10_ERR']
        y = sub21['CO21']
        y_err = sub21['CO21_ERR']
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
                                        logprob2d_scatter_mixture,
                                        args=[x,y,x_err,y_err])
        pos, prob, state = sampler.run_mcmc(p0, 500)
        sampler.reset()
        sampler.run_mcmc(pos,2000)
        sampler_plot2d(sampler,name=name,suffix='.21')

        t['phi'][ctr] = np.median(sampler.flatchain[:,0])
        t['phi+'][ctr] = scipy.stats.scoreatpercentile(\
            sampler.flatchain[:,0],85) - t['phi'][ctr]
        t['phi-'][ctr] = t['phi'][ctr] - \
            scipy.stats.scoreatpercentile(
                sampler.flatchain[:,0],15)

        r21 = np.tan(sampler.flatchain[:,0])

        t['xR21'][ctr] = np.median(r21)
        t['xR21+'][ctr] = scipy.stats.scoreatpercentile(r21,85)
        t['xR21-'][ctr] = scipy.stats.scoreatpercentile(r21,15)

        t['var21'][ctr] = np.median(sampler.flatchain[:,1]**2+
                sampler.flatchain[:,2]**2)

        t['xoff'][ctr] = np.median(sampler.flatchain[:,3])
    if len(sub32)>1:

        x = sub32['CO21']
        x_err = sub32['CO21_ERR']
        y = sub32['CO32']
        y_err = sub32['CO32_ERR']
        ndim, nwalkers = 6,50
        p0 = np.zeros((nwalkers,ndim))
        p0[:,0] = np.random.rand(nwalkers)*np.pi/2
        p0[:,1] = (np.random.randn(nwalkers))**2
        p0[:,2] = (np.random.randn(nwalkers))**2
        p0[:,3] = (np.random.randn(nwalkers))**2*0.05
        p0[:,4] = (np.random.randn(nwalkers))
        p0[:,5] =(np.random.randn(nwalkers))**2*10
        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                        logprob2d_scatter_mixture,
                                        args=[x,y,x_err,y_err])
        pos, prob, state = sampler.run_mcmc(p0, 500)
        sampler.reset()
        sampler.run_mcmc(pos,2000)
        sampler_plot2d(sampler,name=name,suffix='.32')

        t['phi'][ctr] = np.median(sampler.flatchain[:,0])
        t['phi+'][ctr] = scipy.stats.scoreatpercentile(\
            sampler.flatchain[:,0],85) - t['phi'][ctr]
        t['phi-'][ctr] = t['phi'][ctr] - \
            scipy.stats.scoreatpercentile(
                sampler.flatchain[:,0],15)

        r32 = np.tan(sampler.flatchain[:,0])

        t['xR32'][ctr] = np.median(r32)
        t['xR32+'][ctr] = scipy.stats.scoreatpercentile(r32,85)
        t['xR32-'][ctr] = scipy.stats.scoreatpercentile(r32,15)
        t['var32'][ctr] = np.median(sampler.flatchain[:,1]**2+
                sampler.flatchain[:,2]**2)

    it.iternext()

t['var'] = np.sqrt(t['var'])
t.write('brs.txt',format='ascii')
t2 = Table(t)
t2.remove_columns(('theta','phi','theta+','phi+','theta-','phi-'))
emptystring = np.empty((len(t2)),dtype='string')
emptystring[:]=''
col = Column(name='blank',data=emptystring)
t2.add_column(col,index=4)
col = Column(name='blank2',data=emptystring)
t2.add_column(col,index=8)
t2.write('ratios_xoff_scatter.tex',format='latex')

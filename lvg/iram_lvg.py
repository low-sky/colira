import numpy as np
import scipy.stats
import astropy.io.fits as fits
import emcee
import matplotlib.pyplot as p
from astropy.table import Table, Column
from matplotlib import rc
import Radex


execfile('radexlogprob.py')    
    

# Get the specta
s = fits.getdata('iram_spectra.fits')
hdr = fits.getheader('iram_spectra.fits')


# Get the associated data
colira = fits.getdata('colira_subset.fits')
colira_hdr = fits.getheader('colira_subset.fits')

linewidth = 8.0

t = Table(names=('n','n+','n-','N','N+','N-','f','f+','f-','Tk','Tk+','Tk-'))

ctr = 0
for spec in s:

    t.add_row()
    delta = ((colira['RA']-spec['RA'])*np.cos(colira['DEC']))**2+\
        (colira['RA']-spec['RA'])**2
    idx = np.argmin(delta)
    print(np.sqrt(delta[idx])*3600)
    co10 = spec['WCO']
    co10_err = spec['WCO_ERR']
    thirteenco = spec['WCOTOPE']
    thirteenco_err = spec['WCOTOPE_ERR']
    co21 = spec['WHERACON']
    co21_err = spec['WHERACON_ERR']
    co32 = colira['CO32'][idx]*spec['WHERACON']/colira['CO21'][idx]
    co32_err = colira['CO32_ERR'][idx]*spec['WHERACON']/colira['CO21'][idx]
    
    print(co32)
    print(spec['WHERACON']/colira['CO21'][idx])
    print(RadexLogProb([10,4,17,0.2,1/50.], co10 = co10,co21 = co21,co32 = co32,
                       thirteenco = thirteenco,
                       co10_err = co10_err, co21_err = co21_err,
                       co32_err = co32_err,
                       thirteenco_err = thirteenco_err,
                       linewidth = linewidth))

    ndim, nwalkers = 5,100
    p0 = (np.random.rand(ndim*nwalkers).reshape((nwalkers,ndim))*2-1)*0.1+1
    xnull,ynull = np.meshgrid([20,3,17,0.2,1/50.],np.ones(nwalkers))
    p0 = p0*xnull

#p0 = np.array([10,4,17,0.2,1/50.])#*(1+(2*np.random.rand(ndim)-1)*0.2)
#print(p0)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, RadexLogProb, threads=10, \
                                        args=[co10,co21,co32,thirteenco,co10_err,\
                                                  co21_err,co32_err,thirteenco_err,\
                                                  linewidth])
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(pos, 1000)

    t['Tk'][ctr] = np.median(sampler.flatchain[:,0])
    t['Tk+'][ctr] = scipy.stats.scoreatpercentile(sampler.flatchain[:,0],85)
    t['Tk-'][ctr] = scipy.stats.scoreatpercentile(sampler.flatchain[:,0],15)


    t['n'][ctr] = np.median(sampler.flatchain[:,1])
    t['n+'][ctr] = scipy.stats.scoreatpercentile(sampler.flatchain[:,1],85)
    t['n-'][ctr] = scipy.stats.scoreatpercentile(sampler.flatchain[:,1],15)

    t['N'][ctr] = np.median(sampler.flatchain[:,2])
    t['N+'][ctr] = scipy.stats.scoreatpercentile(sampler.flatchain[:,2],85)
    t['N-'][ctr] = scipy.stats.scoreatpercentile(sampler.flatchain[:,2],15)

    t['f'][ctr] = np.median(sampler.flatchain[:,3])
    t['f+'][ctr] = scipy.stats.scoreatpercentile(sampler.flatchain[:,3],85)
    t['f-'][ctr] = scipy.stats.scoreatpercentile(sampler.flatchain[:,3],15)

    ctr = ctr+1

t.write('lvg.txt',format='ascii')

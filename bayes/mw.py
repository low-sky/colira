from . import logprob as lp
from . import sampler_plot as splt
from . import brs as brs
import scipy.stats as ss
import numpy as np
import astropy.wcs as wcs
import astropy.io.fits as fits
import emcee
import matplotlib.pyplot as p
from matplotlib import rc
from astropy.table import Table, Column
import pdb
import sys

def mw(co10file='planck_co10.fits',
       co21file='planck_co21.fits',
       co32file='planck_co32.fits', threads=1):

    co10 = fits.getdata(co10file)
    co10_err = fits.getdata(co10file,1)
    co21 = fits.getdata(co21file)
    co21_err = fits.getdata(co21file,1)
    co32 = fits.getdata(co32file)
    co32_err = fits.getdata(co32file,1)
    w = wcs.WCS(co10file)
    y,x = np.meshgrid(np.arange(co10.shape[0]),
                      np.arange(co10.shape[1]),indexing='ij')

    l,b = w.wcs_pix2world(x,y,0)
# no wrap
    l[l>180] -= 360
    co10 = co10.ravel()
    co21 = co21.ravel()
    co32 = co32.ravel()
    co10_err = co10_err.ravel()
    co21_err = co21_err.ravel()
    co32_err = co32_err.ravel()
    l = l.ravel()
    b = b.ravel()
    t = brs.table_template()
    for tag in t.keys():
        if tag != 'Name':
            t[tag].format = '{:.3f}'
    bins = np.digitize(l.ravel(),np.linspace(l.min(),l.max(),18))
    ubins = np.unique(bins)
    for thisbin in ubins:
        t.add_row()
        idx = np.where((bins == thisbin)*(np.abs(b)<1))
        name = 'MW.{0}'.format(np.min(l[idx]))
        x = co10[idx]
        x_err = co10_err[idx]
        y = co21[idx]
        y_err = co21_err[idx]
        t['Npts'][-1]=x.size
        data = dict(x=x,x_err=x_err,y=y,y_err=y_err)
        ndim, nwalkers = 6,50
        p0 = np.zeros((nwalkers,ndim))
        p0[:,0] = np.pi/6+np.random.randn(nwalkers)*np.pi/8
        p0[:,1] = (np.random.randn(nwalkers))**2*(np.median(x_err)**2+np.median(y_err)**2) # scatter
        p0[:,2] = (np.random.randn(nwalkers)*0.01)**2 # bad fraction
        p0[:,3] = np.percentile(x,50)+np.random.randn(nwalkers)*np.median(x_err)
        p0[:,4] = np.percentile(y,50)+np.random.randn(nwalkers)*np.median(y_err)
        p0[:,5] = np.percentile(x,90)+np.median(x_err)*np.random.randn(nwalkers)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp.logprob2d_scatter_mixture,
                                        args=[x,y,x_err,y_err],threads=threads)
        pos, prob, state = sampler.run_mcmc(p0, 400)
        sampler.reset()
        sampler.run_mcmc(pos,1000)

        print('Name {0}, Acceptance Fraction {1}, Ratio {2}'.format(name,np.mean(sampler.acceptance_fraction),
                                                                                 np.tan(np.median(sampler.flatchain[:,0]))))
        badprob = brs.logprob2d_checkbaddata(sampler,x,y,x_err,y_err)
        splt.sampler_plot2d_mixture(sampler,data,name=name+'.21',badprob=badprob,type='r21')
        brs.summarize2d(t,sampler21=sampler)


        x = co21[idx]
        x_err = co21_err[idx]
        y = co32[idx]
        y_err = co32_err[idx]
        t['Npts'][-1]=x.size
        data = dict(x=x,x_err=x_err,y=y,y_err=y_err)

        ndim, nwalkers = 6,50
        p0 = np.zeros((nwalkers,ndim))
        p0[:,0] = np.pi/6+np.random.randn(nwalkers)*np.pi/8
        p0[:,1] = (np.random.randn(nwalkers))**2*(np.median(x_err)**2+np.median(y_err)**2) # scatter
        p0[:,2] = (np.random.randn(nwalkers)*0.01)**2 # bad fraction
        p0[:,3] = np.percentile(x,50)+np.random.randn(nwalkers)*np.median(x_err)
        p0[:,4] = np.percentile(y,50)+np.random.randn(nwalkers)*np.median(y_err)
        p0[:,5] = np.percentile(x,90)+np.median(x_err)*np.random.randn(nwalkers)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp.logprob2d_scatter_mixture,
                                        args=[x,y,x_err,y_err],threads=threads)
        pos, prob, state = sampler.run_mcmc(p0, 400)
        sampler.reset()
        sampler.run_mcmc(pos,1000)
        print('Name {0}, Acceptance Fraction {1}, Ratio {2}'.format(name,np.mean(sampler.acceptance_fraction),
                                                                                 np.tan(np.median(sampler.flatchain[:,0]))))
        badprob = brs.logprob2d_checkbaddata(sampler,x,y,x_err,y_err)
        splt.sampler_plot2d_mixture(sampler,data,name=name+'.32',badprob=badprob,type='r32')
        brs.summarize2d(t,sampler32=sampler)
        t.write('brs.mw2d.txt',format='ascii')

    
        


    


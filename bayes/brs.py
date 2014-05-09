from . import logprob as lp
from . import sampler_plot as splt
import scipy.stats
import numpy as np
import astropy.io.fits as fits
import emcee
import matplotlib.pyplot as p
from matplotlib import rc
from astropy.table import Table, Column
import pdb
rc('text',usetex=True)

def summarize(t,sampler):
    """
    Summarizes a sampler behavior into the output table

    """

    t['theta'][-1] = np.median(sampler.flatchain[:,0])
    t['theta+'][-1] = scipy.stats.scoreatpercentile(\
        sampler.flatchain[:,0],85) - t['theta'][-1]
    t['theta-'][-1] = t['theta'][-1] - \
        scipy.stats.scoreatpercentile(\
        sampler.flatchain[:,0],15)
    t['phi'][-1] = np.median(sampler.flatchain[:,1])
    t['phi+'][-1] = scipy.stats.scoreatpercentile(\
        sampler.flatchain[:,1],85) - t['phi'][-1]
    t['phi-'][-1] = t['phi'][-1] - \
        scipy.stats.scoreatpercentile(
            sampler.flatchain[:,1],15)

    r21 = np.tan(sampler.flatchain[:,1])
    t['R21'][-1] = np.median(r21)
    t['R21+'][-1] = scipy.stats.scoreatpercentile(r21,85)
    t['R21-'][-1] = scipy.stats.scoreatpercentile(r21,15)


    r32 = 1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1])
    t['R32'][-1] = np.median(r32)
    t['R32+'][-1] = scipy.stats.scoreatpercentile(r32,85)
    t['R32-'][-1] = scipy.stats.scoreatpercentile(r32,15)

    r31 = 1/np.tan(sampler.flatchain[:,0])/np.cos(sampler.flatchain[:,1])
    t['R31'][-1] = np.median(r31)
    t['R31+'][-1] = scipy.stats.scoreatpercentile(r31,85)
    t['R31-'][-1] = scipy.stats.scoreatpercentile(r31,15)
    t['var'][-1] = np.median(sampler.flatchain[:,2]**2+
            sampler.flatchain[:,3]**2+
            sampler.flatchain[:,4]**2)

def table_template():
    t = Table(names=('Name','theta','theta+','theta-','phi','phi+','phi-',\
                     'R21-','R21','R21+','R32-','R32','R32+',
                     'xR21-','xR21','xR21+','xR32-','xR32','xR32+',
                     'var32','var21',
                     'R31-','R31','R31+','var','xoff','LowKey',\
                     'HighKey','MedKey','Npts','Npts21','Npts32'),\
              dtypes=('S7','f8','f8','f8','f8','f8','f8','f8','f8','f8',\
                          'f8','f8','f8','f8','f8','f8','f8','f8',\
                          'f8','f8','f8','f8','f8','f8','f8','f8',\
                          'f8','f8','f8','f8','f8','f8'))
    return t

def bygal(fitsfile):
    s = fits.getdata(fitsfile)
    hdr = fits.getheader(fitsfile)
    GalNames = np.unique(s['GALNAME'])
    
    cut = -10

    t = table_template()
    for tag in t.keys():
        if tag != 'Name':
            t[tag].format = '{:.3f}'

    nGal = len(GalNames)
    it = np.nditer(GalNames,flags=['f_index'])

    while not it.finished:
        ctr = it.index
        name = np.array_str(it.value)
        t.add_row()
        t['Name'][-1] = name.upper()
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
            t['Npts'][-1]=x.size
            data = dict(x=x,x_err=x_err,y=y,y_err=y_err,z=z,z_err=z_err)
            ndim, nwalkers = 5,50
            p0 = np.zeros((nwalkers,ndim))
            p0[:,0] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
            p0[:,1] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
            p0[:,2] = (np.random.randn(nwalkers))**2
            p0[:,3] = (np.random.randn(nwalkers))**2
            p0[:,4] = (np.random.randn(nwalkers))**2
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lp.logprob3d, 
                                        args=[x,y,z,x_err,y_err,z_err])
            pos, prob, state = sampler.run_mcmc(p0, 400)
            sampler.reset()
            sampler.run_mcmc(pos,1000)
            print(name,np.mean(sampler.acceptance_fraction),
                  1/np.tan(np.median(sampler.flatchain[:,0])))

            splt.sampler_plot(sampler,data,name=name)
            summarize(t,sampler)
    it.iternext()

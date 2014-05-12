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

        idx = np.where(
            (s['GALNAME']==name)&
            (s['CO10']>cut*s['CO10_err'])&
            (s['CO10']>cut*s['CO10_err'])&
            (s['CO10']>cut*s['CO10_err'])&
            (s['SPIRE1']> 10.0))
        sub = s[idx]

        idx21 = np.where((s['GALNAME']==name)&
                         (s['CO10']>cut*s['CO10_ERR'])&
                         (s['CO21']>cut*s['CO21_ERR'])&
                         (s['SPIRE1']> 1.0))
        sub21 = s[idx21]

        idx32 = np.where((s['GALNAME']==name)&
                         (s['CO32']>cut*s['CO32_ERR'])&
                         (s['CO21']>cut*s['CO21_ERR']))

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

def bycategory(fitsfile,category=['RGAL','SPIRE1','RGALNORM','FUV',
                                  'UVCOLOR','SFR','IRCOLOR',
                                  'STELLARSD','MOLRAT','PRESSURE']):
    category = np.array(category)
    s = fits.getdata(fitsfile)
    hdr = fits.getheader(fitsfile)
    GalNames = np.unique(s['GALNAME'])

    cut = -10
    quantile = 10
    nValid = 1

    dq = 1e2/quantile
    lower_percentiles = np.arange(quantile)*dq
    upper_percentiles = np.arange(quantile)*dq+dq

    it = np.nditer(lower_percentiles,flags=['f_index'])
    t=table_template()
    nGal = len(GalNames)

    # Identify significant emission on keys

    SignifData = ((s['CO10']>cut*s['CO10_err'])&
                  (s['CO10']>cut*s['CO10_err'])&
                  (s['CO10']>cut*s['CO10_err'])&
                  (s['INTERF']==0)&
                  (s['SPIRE1']> 10.0))
    sub = s[SignifData]

    Signif21 = ((s['CO10']>cut*s['CO10_ERR'])&\
                (s['CO21']>cut*s['CO21_ERR'])&\
                (s['INTERF']==0)&\
                (s['SPIRE1']> 1.0))
    sub21 = s[Signif21]

    Signif32 = ((s['CO32']>cut*s['CO32_ERR'])&\
                (s['CO21']>cut*s['CO21_ERR'])&\
                (s['INTERF']==0))
    sub32 = s[Signif32]
    
    molrat = 313*s['CO21']/s['HI']
    sfr = 634*s['HA']+0.00325*s['MIPS24']
    ircolor = (s['MIPS24']/s['PACS3'])
    stellarsd = 200*s['IRAC1']
    pressure = 272*(s['HI']*0.02+s['CO21']*6.7)*\
               np.sqrt(stellarsd)*8/np.sqrt(212)


    iter2 = np.nditer(category)

    while not iter2.finished:
        keyname = str(iter2.value)
        if keyname == 'SFR':
            key_variable = sfr
        elif keyname == 'IRCOLOR':
            key_variable = ircolor
        elif keyname == 'STELLARSD':
            key_variable = stellarsd
        elif keyname == 'MOLRAT':
            key_variable = molrat
        elif keyname == 'PRESSURE':
            key_variable = pressure
        elif keyname == 'RGALNORM':
            key_variable = s['RGALNORM']
        elif keyname == 'RGAL':
            key_variable = s['RGAL']
        elif keyname == 'FUV':
            key_variable = s['GALEXFUV']
        elif keyname == 'UVCOLOR':
            key_variable = (s['GALEXFUV']/s['GALEXNUV'])
        elif keyname == 'SPIRE1':
            key_variable = (s['SPIRE1'])

        it = np.nditer(lower_percentiles,flags=['f_index'])
        print(keyname)
        keyscores = key_variable[np.isfinite(key_variable)&SignifData]
        t=table_template()
        while not it.finished:
            pct = it.index
            name = np.array_str(it.value)
            t.add_row()
            t['Name'][pct] = name.upper()

            lower_score =scipy.stats.scoreatpercentile(keyscores,\
                                                           lower_percentiles[pct])
            upper_score =scipy.stats.scoreatpercentile(keyscores,\
                                                           upper_percentiles[pct])

            t['LowKey'][pct] = np.log10(lower_score)
            t['HighKey'][pct] = np.log10(upper_score)
            print(lower_score,upper_score)
            idx = np.where((key_variable>=lower_score)&
                           (key_variable<=upper_score)&(SignifData))
            sub = s[idx]
            t['MedKey'][pct] =np.log10(np.median(key_variable[idx]))

            idx21 = np.where((key_variable>=lower_score)&
                           (key_variable<=upper_score)&(Signif21))
            idx32 = np.where((key_variable>=lower_score)&
                           (key_variable<=upper_score)&(Signif32))

            sub21 = s[idx21]
            sub32 = s[idx32]
            print(len(sub21),len(sub32),len(sub))

            if len(sub)>nValid:
                x = sub['CO10']
                x_err = sub['CO10_ERR']
                y = sub['CO21']
                y_err = sub['CO21_ERR']
                z = sub['CO32']
                z_err = sub['CO32_ERR']
                ndim, nwalkers = 5,50
                p0 = np.zeros((nwalkers,ndim))

                p0[:,0] = np.pi/2*np.random.rand(nwalkers)
                p0[:,1] = np.pi/2*np.random.rand(nwalkers)
                p0[:,2] = (np.random.randn(nwalkers))**2
                p0[:,3] = (np.random.randn(nwalkers))**2
                p0[:,4] = (np.random.randn(nwalkers))**2

                sampler = emcee.EnsembleSampler(nwalkers, ndim, lp.logprob3d, 
                                            args=[x,y,z,x_err,y_err,z_err])
                pos, prob, state = sampler.run_mcmc(p0, 200)
                sampler.reset()
                sampler.run_mcmc(pos, 1000)

                splt.sampler_plot(sampler,data,name=name)
                summarize(t,sampler)
            it.iternext()
    iter2.iternext()

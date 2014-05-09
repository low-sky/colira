#!/usr/bin/env python

import scipy.stats
import numpy as np
import astropy.io.fits as fits
import emcee
import matplotlib.pyplot as p
from matplotlib import rc
from astropy.table import Table, Column
from matplotlib.backends.backend_pdf import PdfPages
#rc('text',usetex=True)

execfile('logprob.py')
execfile('sampler_plot.py')
s = fits.getdata('colira.fits')
hdr = fits.getheader('colira.fits')
GalNames = np.unique(s['GALNAME'])
figdir ='../plots/'

cut = -10
quantile = 20
nValid = 1

dq = 5e2
#lower_percentiles = np.arange(quantile)*dq
#upper_percentiles = np.arange(quantile)*dq+dq

lower_value = np.arange(quantile)*dq
upper_value = np.arange(quantile)*dq+dq

it = np.nditer(lower_value,flags=['f_index'])

NewTable = Table(names=('Name','theta','theta+','theta-','phi','phi+','phi-',\
                     'R21-','R21','R21+','R32-','R32','R32+',
                     'xR21-','xR21','xR21+','xR32-','xR32','xR32+',
                     'var32','var21',
                     'R31-','R31','R31+','var','xoff','LowKey',\
                     'HighKey','MedKey','Npts','Npts21','Npts32'),\
              dtypes=('S7','f8','f8','f8','f8','f8','f8','f8','f8','f8',\
                          'f8','f8','f8','f8','f8','f8','f8','f8',\
                          'f8','f8','f8','f8','f8','f8','f8','f8',\
                          'f8','f8','f8','f8','f8','f8'))

for tag in NewTable.keys():
    if tag != 'Name':
        NewTable[tag].format = '{:.3f}'

nGal = len(GalNames)

# Identify significant emission on keys


SignifData  = np.logical_and(np.logical_and((s['CO10']>cut*s['CO10_ERR']), 
                                            (s['CO21']>cut*s['CO21_ERR'])),\
                                 (s['CO32']>cut*s['CO32_ERR']))
SignifData = (s['CO10']>cut*s['CO10_ERR'])*(s['CO21']>cut*s['CO21_ERR'])*(s['CO32']>cut*s['CO32_ERR'])*(s['INTERF']==0)

Signif21  = np.logical_and((s['CO10']>cut*s['CO10_ERR'])*(s['INTERF']==0),\
                           (s['CO21']>cut*s['CO21_ERR']))

Signif32 = np.logical_and((s['CO32']>cut*s['CO32_ERR']),
                          (s['CO21']>cut*s['CO21_ERR']))

molrat = 313*s['CO21']/s['HI']
sfr = 634*s['HA']+0.00325*s['MIPS24']
ircolor = (s['MIPS24']/s['PACS3'])
stellarsd = 200*s['IRAC1']
pressure = 272*(s['HI']*0.02+s['CO21']*6.7)*np.sqrt(stellarsd)*8/np.sqrt(212)

# KeyArray = np.array(['RGAL','SPIRE1','RGALNORM','FUV',
#                      'UVCOLOR','SFR','IRCOLOR',
#                      'STELLARSD','MOLRAT','PRESSURE'])

KeyArray = np.array(['RGAL'])
iter2 = np.nditer(KeyArray)

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
    it = np.nditer(lower_value,flags=['f_index'])
    print(keyname)
    keyscores = key_variable[np.isfinite(key_variable)&SignifData]
    t=Table(NewTable,copy=True)
    while not it.finished:
        pct = it.index
        name = np.array_str(it.value)
        t.add_row()
        t['Name'][pct] = name.upper()

        lower_score = lower_value[pct]
                                     
        upper_score = upper_value[pct]
        scorestr = str(lower_score)

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
            
            sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob3d, 
                                        args=[x,y,z,x_err,y_err,z_err])
            pos, prob, state = sampler.run_mcmc(p0, 200)
            sampler.reset()
            sampler.run_mcmc(pos, 1000)

#            sampler_plot(sampler,name = keyname+'_'+name)

    #        p.savefig(pp,format='pdf',orientation='portrait')

            print(name,np.mean(sampler.acceptance_fraction),
                  1/np.tan(np.median(sampler.flatchain[:,0])))

            t['theta'][pct] = np.median(sampler.flatchain[:,0])
            t['theta+'][pct] = scipy.stats.scoreatpercentile(\
                sampler.flatchain[:,0],85) - t['theta'][pct]
            t['theta-'][pct] = t['theta'][pct] - \
                scipy.stats.scoreatpercentile(\
                sampler.flatchain[:,0],15)
            t['phi'][pct] = np.median(sampler.flatchain[:,1])
            t['phi+'][pct] = scipy.stats.scoreatpercentile(\
                sampler.flatchain[:,1],85) - t['phi'][pct]
            t['phi-'][pct] = t['phi'][pct] - \
                scipy.stats.scoreatpercentile(\
                sampler.flatchain[:,1],15)

            r21 = np.tan(sampler.flatchain[:,1])
            t['R21'][pct] = np.median(r21)
            t['R21+'][pct] = scipy.stats.scoreatpercentile(r21,85)
            t['R21-'][pct] = scipy.stats.scoreatpercentile(r21,15)


            r32 = 1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1])
            t['R32'][pct] = np.median(r32)
            t['R32+'][pct] = scipy.stats.scoreatpercentile(r32,85)
            t['R32-'][pct] = scipy.stats.scoreatpercentile(r32,15)

            r31 = 1/np.tan(sampler.flatchain[:,0])/np.cos(sampler.flatchain[:,1])
            t['R31'][pct] = np.median(r31)
            t['R31+'][pct] = scipy.stats.scoreatpercentile(r31,85)
            t['R31-'][pct] = t['R31'][pct] =scipy.stats.scoreatpercentile(r31,15)
            t['var'][pct] = np.median(sampler.flatchain[:,2]**2+
                                      sampler.flatchain[:,3]**2+
                                      sampler.flatchain[:,4]**2)
            t['Npts'][pct] = len(sub)

        if len(sub21)>nValid:
            x = sub21['CO10']
            x_err = sub21['CO10_ERR']
            y = sub21['CO21']
            y_err = sub21['CO21_ERR']
            ndim, nwalkers = 6,50
            p0 = np.zeros((nwalkers,ndim))
            p0[:,0] = np.random.rand(nwalkers)*np.pi/2
            p0[:,1] = (np.random.randn(nwalkers))**2
            p0[:,2] = (np.random.randn(nwalkers))**2
            p0[:,3] = (np.random.randn(nwalkers))**2*0.05
            p0[:,4] = (np.random.randn(nwalkers))
            p0[:,5] =(np.random.randn(nwalkers))**2*10

            sampler = emcee.EnsembleSampler(nwalkers, ndim, \
                                            logprob2d_scatter_mixture,
                                            args=[x,y,x_err,y_err])
            pos, prob, state = sampler.run_mcmc(p0, 400)
            sampler.reset()
            sampler.run_mcmc(pos,1000)
            sampler_plot2d(sampler,name = keyname+'_fixbin_'+scorestr,
                           suffix ='.21')

            t['phi'][pct] = np.median(sampler.flatchain[:,0])
            t['phi+'][pct] = scipy.stats.scoreatpercentile(\
                sampler.flatchain[:,0],85) - t['phi'][pct]
            t['phi-'][pct] = t['phi'][pct] - \
                scipy.stats.scoreatpercentile(
                    sampler.flatchain[:,0],15)

            r21 = np.tan(sampler.flatchain[:,0])

            t['xR21'][pct] = np.median(r21)
            t['xR21+'][pct] = scipy.stats.scoreatpercentile(r21,85)
            t['xR21-'][pct] = scipy.stats.scoreatpercentile(r21,15)

            t['var21'][pct] = np.median(sampler.flatchain[:,1]**2+
                    sampler.flatchain[:,2]**2)
            t['Npts21'][-1] = len(sub21)
        if len(sub32)>nValid:

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

    #        p0[:,0] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
    #        p0[:,1] = (np.random.randn(nwalkers))**2
    #        p0[:,2] = (np.random.randn(nwalkers))**2
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                            logprob2d_scatter_mixture, 
                                            args=[x,y,x_err,y_err])
            pos, prob, state = sampler.run_mcmc(p0, 400)
            sampler.reset()
            sampler.run_mcmc(pos,1000)
            sampler_plot2d(sampler,xoff=0,name =keyname+'_fixbin_'+scorestr,
                           suffix = '.32')

            t['phi'][pct] = np.median(sampler.flatchain[:,0])
            t['phi+'][pct] = scipy.stats.scoreatpercentile(\
                sampler.flatchain[:,0],85) - t['phi'][pct]
            t['phi-'][pct] = t['phi'][pct] - \
                scipy.stats.scoreatpercentile(
                    sampler.flatchain[:,0],15)

            r32 = np.tan(sampler.flatchain[:,0])

            t['xR32'][pct] = np.median(r32)
            t['xR32+'][pct] = scipy.stats.scoreatpercentile(r32,85)
            t['xR32-'][pct] = scipy.stats.scoreatpercentile(r32,15)
            t['var32'][pct] = np.median(sampler.flatchain[:,1]**2+
                    sampler.flatchain[:,2]**2)
            t['Npts32'][-1] = len(sub32)
        it.iternext()
    t.write('brs_fixradialbin.'+keyname+'.txt',format='ascii')
    t2 = Table(t,copy=True)
    t2.remove_columns(('theta','phi','theta+','phi+','theta-','phi-'))
    emptystring = np.empty((len(t2)),dtype='string')
    emptystring[:]=''
    col = Column(name='blank',data=emptystring)
    t2.add_column(col,index=4)
    col = Column(name='blank2',data=emptystring)
    t2.add_column(col,index=8)
    t2.write('brs_fixradialbin.'+keyname+'.tex',format='latex')
    iter2.iternext()
#pp.close()


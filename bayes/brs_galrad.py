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
s = fits.getdata('colira_fixedresn.fits')
hdr = fits.getheader('colira_fixedresn.fits')
GalNames = np.unique(s['GALNAME'])
figdir ='../plots/'

cut = -10
quantile = 10
nValid = 1 # Number of points to consider a valid fit
dq = 1e0/quantile
lower_values = np.arange(quantile)*dq
upper_values = np.arange(quantile)*dq+dq



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
for tag in t.keys():
    if tag != 'Name':
        t[tag].format = '{:.3f}'

nGal = len(GalNames)
it = np.nditer(lower_values,flags=['f_index'])

# Identify significant emission on keys


SignifData  = np.logical_and(np.logical_and((s['CO10']>cut*s['CO10_ERR']), 
                                            (s['CO21']>cut*s['CO21_ERR'])),\
                                 (s['CO32']>cut*s['CO32_ERR']))


Signif21  = np.logical_and((s['CO10']>cut*s['CO10_ERR']),\
                           (s['CO21']>cut*s['CO21_ERR']))

Signif32 = np.logical_and((s['CO32']>cut*s['CO32_ERR']),
                          (s['CO21']>cut*s['CO21_ERR']))

keyname = 'RGALNORM'
key_variable = s['RGALNORM']

#pp = PdfPages(figdir+keyname+'.pdf')

iter2 = np.nditer(GalNames,flags=['f_index'])

while not iter2.finished:
    name = np.array_str(iter2.value)
    it = np.nditer(lower_values,flags=['f_index'])
    while not it.finished:
        pct = it.index
        t.add_row()
        t['Name'][-1] = name

        t['LowKey'][-1] = lower_values[pct]
        t['HighKey'][-1] = upper_values[pct]
        print(name,lower_values[pct],upper_values[pct])
        idx = np.where((key_variable>=lower_values[pct])&
                       (key_variable<=upper_values[pct])&(SignifData)&(name==s['GALNAME']))
        sub = s[idx]
        t['MedKey'][-1] =np.median(key_variable[idx])

        idx21 = np.where((key_variable>=lower_values[pct])&
                       (key_variable<=upper_values[pct])&(Signif21)&(name==s['GALNAME']))
        idx32 = np.where((key_variable>=lower_values[pct])&
                        (key_variable<=upper_values[pct])&(Signif32)&(name==s['GALNAME']))

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
                scipy.stats.scoreatpercentile(\
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
            t['R31-'][-1] = t['R31'][-1] =scipy.stats.scoreatpercentile(r31,15)
            t['var'][-1] = np.median(sampler.flatchain[:,2]**2+
                                      sampler.flatchain[:,3]**2+
                                      sampler.flatchain[:,4]**2)
            t['Npts'][-1] = len(sub)

        if len(sub21)>nValid:
            x = sub21['CO10']
            x_err = sub21['CO10_ERR']
            y = sub21['CO21']
            y_err = sub21['CO21_ERR']
            ndim, nwalkers = 7,50
            p0 = np.zeros((nwalkers,ndim))
    #        p0[:,0] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
    #        p0[:,1] = (np.random.randn(nwalkers))**2
    #        p0[:,2] = (np.random.randn(nwalkers))**2

            p0[:,0] = np.random.rand(nwalkers)*np.pi/2
            p0[:,1] = (np.random.randn(nwalkers))**2
            p0[:,2] = (np.random.randn(nwalkers))**2
            p0[:,3] = (np.random.randn(nwalkers))
            p0[:,4] = (np.random.randn(nwalkers))**2*0.05
            p0[:,5] = (np.random.randn(nwalkers)) 
            p0[:,6] =(np.random.randn(nwalkers))**2*10

            sampler = emcee.EnsembleSampler(nwalkers, ndim, \
                                            logprob2d_xoff_scatter_mixture,
                                            args=[x,y,x_err,y_err])
            pos, prob, state = sampler.run_mcmc(p0, 400)
            sampler.reset()
            sampler.run_mcmc(pos,1000)
#            sampler_plot2d(sampler,name = keyname+'_'+name,suffix ='.21')

            t['phi'][-1] = np.median(sampler.flatchain[:,0])
            t['phi+'][-1] = scipy.stats.scoreatpercentile(\
                sampler.flatchain[:,0],85) - t['phi'][-1]
            t['phi-'][-1] = t['phi'][-1] - \
                scipy.stats.scoreatpercentile(
                    sampler.flatchain[:,0],15)

            r21 = np.tan(sampler.flatchain[:,0])

            t['xR21'][-1] = np.median(r21)
            t['xR21+'][-1] = scipy.stats.scoreatpercentile(r21,85)
            t['xR21-'][-1] = scipy.stats.scoreatpercentile(r21,15)

            t['var21'][-1] = np.median(sampler.flatchain[:,1]**2+
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
#            sampler_plot2d(sampler,xoff=0,name =keyname+'_'+name,suffix = '.32')

            t['phi'][-1] = np.median(sampler.flatchain[:,0])
            t['phi+'][-1] = scipy.stats.scoreatpercentile(\
                sampler.flatchain[:,0],85) - t['phi'][-1]
            t['phi-'][-1] = t['phi'][-1] - \
                scipy.stats.scoreatpercentile(
                    sampler.flatchain[:,0],15)

            r32 = np.tan(sampler.flatchain[:,0])

            t['xR32'][-1] = np.median(r32)
            t['xR32+'][-1] = scipy.stats.scoreatpercentile(r32,85)
            t['xR32-'][-1] = scipy.stats.scoreatpercentile(r32,15)
            t['var32'][-1] = np.median(sampler.flatchain[:,1]**2+
                    sampler.flatchain[:,2]**2)
            t['Npts32'][-1] = len(sub32)
        print(it.value,iter2.value)
        it.iternext()
    iter2.iternext()

#pp.close()

t['var'] = np.sqrt(t['var'])
t.write('brs_category.'+keyname+'.txt',format='ascii')
t2 = Table(t)
t2.remove_columns(('theta','phi','theta+','phi+','theta-','phi-'))
emptystring = np.empty((len(t2)),dtype='string')
emptystring[:]=''
col = Column(name='blank',data=emptystring)
t2.add_column(col,index=4)
col = Column(name='blank2',data=emptystring)
t2.add_column(col,index=8)
t2.write('brs_category.'+keyname+'.tex',format='latex')

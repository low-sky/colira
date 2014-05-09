#!/usr/bin/env python

import scipy.stats
import numpy as np
import astropy.io.fits as fits
import emcee
import matplotlib.pyplot as p
from matplotlib import rc
from astropy.table import Table, Column
from matplotlib.backends.backend_pdf import PdfPages
rc('text',usetex=True)

execfile('logprob.py')

s = fits.getdata('colira_fixedresn.fits')
hdr = fits.getheader('colira_fixedresn.fits')
GalNames = np.unique(s['GALNAME'])
figdir ='../plots/'

cut = 1
quantile = 10

dq = 1e2/quantile
lower_percentiles = np.arange(quantile)*dq
upper_percentiles = np.arange(quantile)*dq+dq

it = np.nditer(lower_percentiles,flags=['f_index'])

t = Table(names=('Name','theta','theta+','theta-','phi','phi+','phi-',\
                     'R21-','R21','R21+','R32-','R32','R32+',\
                     'R31-','R31','R31+','var','xoff','LowKey','HighKey','MedKey','Npts'),\
              dtypes=('S7','f8','f8','f8','f8','f8','f8','f8','f8','f8',\
                          'f8','f8','f8','f8','f8','f8','f8','f8','f8',\
                          'f8','f8','f8'))
for tag in t.keys():
    if tag != 'Name':
        t[tag].format = '{:.3f}'

nGal = len(GalNames)

quantile = 10
lower_percentiles = np.arange(quantile)*quantile
upper_percentiles = np.arange(quantile)*quantile+1e2/quantile


it = np.nditer(GalNames,flags=['f_index'])

# Identify significant emission on keys


SignifData  = np.logical_and(np.logical_and((s['CO10']>cut*s['CO10_ERR']), 
                                            (s['CO21']>cut*s['CO21_ERR'])),\
                                 (s['CO32']>cut*s['CO32_ERR']))

molrat = 313*s['CO21']/s['HI']
sfr = 634*s['HA']+0.00325*s['MIPS24']
keyname='RGALNORM'
key_variable = s['RGALNORM']


pp = PdfPages(keyname+'.pdf')

while not it.finished:
    for radidx in np.arange(len(lower_percentiles)):
        pct = it.index
        name = np.array_str(it.value)
        t.add_row()
        t['Name'][-1] = name.upper()
        keyscores = key_variable[np.isfinite(key_variable)&SignifData]
        lower_score =scipy.stats.scoreatpercentile(keyscores,\
                                                       lower_percentiles[radidx])
        upper_score =scipy.stats.scoreatpercentile(keyscores,\
                                                       upper_percentiles[radidx])

        t['LowKey'][-1] = lower_score
        t['HighKey'][-1] = upper_score
        print(name,lower_percentiles[radidx],lower_score,upper_score)
        idx = np.where((key_variable>=lower_score)&(key_variable<=upper_score)&\
                           (SignifData)&(s['GALNAME']==name))
        sub = s[idx]
        t['MedKey'][-1] =np.median(key_variable[idx])

        if len(sub)>1:
            x = sub['CO10']
            x_err = sub['CO10_ERR']
            y = sub['CO21']
            y_err = sub['CO21_ERR']
            z = sub['CO32']
            z_err = sub['CO32_ERR']
            ndim, nwalkers = 4,50
            p0 = np.zeros((nwalkers,ndim))
            p0[:,0] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
            p0[:,1] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
            p0[:,2] = (np.random.randn(nwalkers))
            p0[:,3] = np.abs((np.random.randn(nwalkers)))*2
            sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob3d_xoff_scatter, 
                                            args=[x,y,z,x_err,y_err,z_err])
            pos, prob, state = sampler.run_mcmc(p0, 200)
            sampler.reset()
            sampler.run_mcmc(pos, 1000)

    #######################
    # PLOTTING
    #######################
            p.figure(figsize=(6,8))
            p.subplot(321)
            p.hist(1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1]),\
                       range=[0,2],bins=100)
            p.xlabel('$R_{32}$')
            p.subplot(322)
            p.hist(np.tan(sampler.flatchain[:,1]),range=[0,2],bins=100)
            p.xlabel('$R_{21}$')

            p.subplot(323)
            p.errorbar(x,y,xerr=x_err,yerr=y_err,fmt=None,marker=None,mew=0)
            p.scatter(x,y,marker='.')
            p.xlabel('CO(1-0)')
            p.ylabel('CO(2-1)')
            testx = np.linspace(np.nanmin(x),np.nanmax(x),10)
            xoff = np.median(sampler.flatchain[:,2])
            p.plot(testx,np.tan(np.median(sampler.flatchain[:,1]))*(testx+xoff),color='r')

            p.subplot(324)
            p.errorbar(y,z,xerr=y_err,yerr=y_err,fmt=None,marker=None,mew=0)
            p.scatter(y,z,marker='.')
            p.xlabel('CO(2-1)')
            p.ylabel('CO(3-2)')
            testx = np.linspace(np.nanmin(y),np.nanmax(y),10)
            p.plot(testx,testx/np.tan(np.median(sampler.flatchain[:,0]))/\
                       np.sin(np.median(sampler.flatchain[:,1])),color='r')

            p.subplot(325)
            p.hexbin(np.tan(sampler.flatchain[:,1]),1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1]))
            p.xlabel('$R_{21}$')
            p.ylabel('$R_{32}$')

            p.subplot(326)
            p.hexbin(np.tan(sampler.flatchain[:,1]),sampler.flatchain[:,2])
            p.xlabel('$R_{21}$')
            p.ylabel('Offset')
#            p.title(keyname+' '+name.upper())
            p.tight_layout()
            p.savefig(pp,format='pdf',orientation='portrait')

            print(name,np.mean(sampler.acceptance_fraction),1/np.tan(np.median(sampler.flatchain[:,0])))

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
#            t['var'][-1] = np.median(sampler.flatchain[:,3])
#            t['xoff'][-1] = np.median(sampler.flatchain[:,2])
            t['var'][ctr] = np.median(sampler.flatchain[:,2]**2+
                                      sampler.flatchain[:,3]**2+
                                      sampler.flatchian[:,4]**2)
            
            t['Npts'][-1] = len(sub)
    it.iternext()
pp.close()

t['var'] = np.sqrt(t['var'])
t.write('brs_category.'+keyname+'.bygal.txt',format='ascii')
t2 = Table(t)
t2.remove_columns(('theta','phi','theta+','phi+','theta-','phi-'))
emptystring = np.empty((len(t2)),dtype='string')
emptystring[:]=''
col = Column(name='blank',data=emptystring)
t2.add_column(col,index=4)
col = Column(name='blank2',data=emptystring)
t2.add_column(col,index=8)
t2.write('brs_category.'+keyname+'.bygal.tex',format='latex')

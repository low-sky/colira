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

s = fits.getdata('colira_subset.fits')
hdr = fits.getheader('colira_subset.fits')
GalNames = np.unique(s['GALNAME'])


t = Table(names=('Name','theta','theta+','theta-','phi','phi+','phi-',\
                     'R21-','R21','R21+','R32-','R32','R32+','var',\
                     'R31-','R31','R31+'),\
              dtypes=('S7','f8','f8','f8','f8','f8','f8','f8','f8','f8',\
                          'f8','f8','f8','f8','f8','f8','f8'))
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
        z = sub['CO32']
        z_err = sub['CO32_ERR']

        ndim, nwalkers = 3,50
        p0 = np.zeros((nwalkers,3))
        p0[:,0] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
        p0[:,1] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
        p0[:,2] = (np.random.random(nwalkers))*20
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob3d, 
                                    args=[x,y,z,x_err,y_err,z_err])
        pos, prob, state = sampler.run_mcmc(p0, 200)
        sampler.reset()
        sampler.run_mcmc(pos, 1000)
        p.figure(1)
#        p.subplot(241)
#Theta
##        p.plot(sampler.flatchain[:,0])
#        p.ylabel(r'$\theta$')

#        p.subplot(242)
#phi
#        p.plot(sampler.flatchain[:,1])
#        p.ylabel(r'$\phi$')

        p.subplot(243)
        p.hist(1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1]),\
                   range=[0,2],bins=100)
        p.xlabel('$R_{32}$')
        p.subplot(244)
        p.hist(np.tan(sampler.flatchain[:,1]),range=[0,2],bins=100)
        p.xlabel('$R_{21}$')
        p.subplot(245)
        p.errorbar(x,y,xerr=x_err,yerr=y_err,fmt=None,marker=None,mew=0)
        p.scatter(x,y,marker='.')
        p.xlabel('CO(1-0)')
        p.ylabel('CO(2-1)')
        testx = np.linspace(np.nanmin(x),np.nanmax(x),10)
        p.plot(testx,np.tan(np.median(sampler.flatchain[:,1]))*testx,color='r')

        p.subplot(246)
        p.errorbar(y,z,xerr=y_err,yerr=y_err,fmt=None,marker=None,mew=0)
        p.scatter(y,z,marker='.')
        p.xlabel('CO(2-1)')
        p.ylabel('CO(3-2)')
        testx = np.linspace(np.nanmin(y),np.nanmax(y),10)
        p.plot(testx,testx/np.tan(np.median(sampler.flatchain[:,0]))/\
                   np.sin(np.median(sampler.flatchain[:,1])),color='r')

        p.subplot(247)
        p.plot(sampler.flatchain[:,2])
        p.xlabel(r'$V$')
        
        p.subplot(248)
        p.hexbin(np.tan(sampler.flatchain[:,1]),\
                     1/np.tan(sampler.flatchain[:,0])/\
                     np.sin(sampler.flatchain[:,1]))
        p.xlabel('$R_{21}$')
        p.ylabel('$R_{32}$')
        p.savefig(name+'.pdf',format='pdf',orientation='portrait')

        p.close()

        print(name,np.mean(sampler.acceptance_fraction),1/np.tan(np.median(sampler.flatchain[:,0])))

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
            scipy.stats.scoreatpercentile(\
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
        t['R31-'][ctr] = t['R31'][ctr] =scipy.stats.scoreatpercentile(r31,15)
        t['var'][ctr] = np.median(sampler.flatchain[:,2])
    it.iternext()

t2 = Table(t)
t2.remove_columns(('theta','phi','theta+','phi+','theta-','phi-','var'))
emptystring = np.empty((len(t2)),dtype='string')
emptystring[:]=''
col = Column(name='blank',data=emptystring)
t2.add_column(col,index=4)
col = Column(name='blank2',data=emptystring)
t2.add_column(col,index=8)
t2.write('ratios.tex',format='latex')

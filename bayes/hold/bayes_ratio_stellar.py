#!/usr/bin/env python

import numpy as np
import scipy.stats
import astropy.io.fits as fits
import emcee
import matplotlib.pyplot as p
from astropy.table import Table, Column
from matplotlib import rc

execfile('logprob.py')



s = fits.getdata('colira_subset.fits')
hdr = fits.getheader('colira_subset.fits')
GalNames = np.unique(s['GALNAME'])

t = Table(names=('Name','theta','theta+','theta-','phi','phi+','phi-',\
                     'R21-','R21','R21+','R32-','R32','R32+',\
                     'R31-','R31','R31+','var','xoff','LowKey','HighKey','MedKey'),\
              dtypes=('S7','f8','f8','f8','f8','f8','f8','f8','f8','f8',\
                          'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8'))
for tag in t.keys():
    if tag != 'Name':
        t[tag].format = '{:.3f}'


quantile=20

dq = 1e2/quantile
lower_percentiles = np.arange(quantile)*dq
upper_percentiles = np.arange(quantile)*dq+dq

stellar = 200*s['IRAC1']
it = np.nditer(lower_percentiles,flags=['f_index'])
it = np.nditer(lower_percentiles,flags=['f_index'])

while not it.finished:
    pct = it.index
    name = np.array_str(it.value)
    t.add_row()
    t['Name'][pct] = name.upper()

    key_variable = stellar
    keyscores = key_variable[np.isfinite(key_variable)]
    lower_score =scipy.stats.scoreatpercentile(keyscores,\
                                                   lower_percentiles[pct])
    upper_score =scipy.stats.scoreatpercentile(keyscores,\
                                                   upper_percentiles[pct])

    t['LowKey'][pct] = lower_score
    t['HighKey'][pct] = upper_score
    print(lower_score,upper_score)
    idx = np.where((key_variable>=lower_score)&(key_variable<=upper_score))
    sub = s[idx]
    t['MedKey'][pct] = np.median(key_variable[idx])
    if len(sub)>1:
        name = 'key_'+(lower_percentiles[pct]).astype('S')
        x = sub['CO10']
        x_err = sub['CO10_ERR']
    
        y = sub['CO21']
        y_err = sub['CO21_ERR']
        z = sub['CO32']
        z_err = sub['CO32_ERR']


        ndim, nwalkers = 4,50
        p0 = np.zeros((nwalkers,4))
        p0[:,0] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
        p0[:,1] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
        p0[:,2] = (np.random.random(nwalkers))*20
        p0[:,3] = (np.random.randn(nwalkers))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob3d_xoff, 
                                    args=[x,y,z,x_err,y_err,z_err])
        pos, prob, state = sampler.run_mcmc(p0, 200)
        sampler.reset()
        sampler.run_mcmc(pos, 1000)

#         p.figure(1)
#         p.subplot(321)
# #Theta
#         p.plot(1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1]))
#         p.subplot(322)
# #phi
#         p.plot(np.tan(sampler.flatchain[:,1]))
#         p.subplot(323)
#         p.hist(1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1]))
#         p.subplot(324)
#         p.hist(np.tan(sampler.flatchain[:,1]))
#         p.subplot(325)
#         p.errorbar(x,y,xerr=x_err,yerr=y_err,fmt=None,marker=None,mew=0)
#         p.scatter(x,y,marker='.')
#         testx = np.linspace(np.nanmin(x),np.nanmax(x),10)
#         p.plot(testx,np.tan(np.median(sampler.flatchain[:,1]))*testx,color='r')
#         p.subplot(326)
#         p.hexbin(1/np.tan(sampler.flatchain[:,0])/\
#                      np.sin(sampler.flatchain[:,1]),\
#                      np.tan(sampler.flatchain[:,1]))
#         p.savefig(name+'.pdf',format='pdf',orientation='portrait')

#         p.close()
        print(name,np.mean(sampler.acceptance_fraction),np.tan(np.median(sampler.flatchain[:,1])))
        t['theta'][pct] = np.median(sampler.flatchain[:,0])
        t['theta+'][pct] = scipy.stats.scoreatpercentile(\
            sampler.flatchain[:,0],85) - t['theta'][pct]
        t['theta-'][pct] = t['theta'][pct] - \
            scipy.stats.scoreatpercentile(sampler.flatchain[:,0],15)
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
        t['var'][pct] = np.median(sampler.flatchain[:,2])
        t['xoff'][pct] = np.median(sampler.flatchain[:,3])
    it.iternext()

t['R21+'] = t['R21+']-t['R21']
t['R21-'] = t['R21']-t['R21-']
t['R32+'] = t['R32+']-t['R32']
t['R32-'] = t['R32']-t['R32-']

fig = p.figure(1,figsize=(4.0,4.0))
ax = p.subplot(111)
ax.set_xscale('log')
ax.set_ylim(0,0.8)
ax.set_xlabel(r'$\Sigma_{\star} (M_{\odot}\mbox{ pc}^{-2})$')
ax.set_ylabel(r'Line Ratio')
p.errorbar(t['MedKey'],t['R32'],yerr=[t['R32-'],t['R32+']],\
               marker='^',color='black',ecolor='gray',label='$R_{32}$')

p.errorbar(t['MedKey'],t['R21'],yerr=[t['R21-'],t['R21+']],\
               marker='o',color='black',ecolor='gray',label='$R_{21}$')
p.tight_layout()
#p.subplots_adjust(bottom=0.14)
p.legend(loc=4)
p.savefig('ratio_vs_stellar.pdf',bbox='tight')
p.close()

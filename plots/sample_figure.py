#!/usr/bin/env python

import scipy.stats
from scipy.interpolate import interp1d
import numpy as np
import astropy.io.fits as fits
import emcee
import matplotlib.pyplot as p
from matplotlib import rc
from astropy.table import Table, Column
rc('text',usetex=True)
rc('font',size=9)
execfile('logprob.py')

s = fits.getdata('colira_subset.fits')
hdr = fits.getheader('colira_subset.fits')
GalNames = np.unique(s['GALNAME'])

idx = np.where(s['GALNAME']=='ngc3351')
sub = s[idx]

x = sub['CO10']
x_err = sub['CO10_ERR']

y = sub['CO21']
y_err = sub['CO21_ERR']
z = sub['CO32']
z_err = sub['CO32_ERR']

ndim, nwalkers = 4,100
p0 = np.zeros((nwalkers,4))
p0[:,0] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
p0[:,1] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
p0[:,2] = (np.random.random(nwalkers))*20
p0[:,3] = (np.random.randn(nwalkers))
sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob3d_xoff, 
                                args=[x,y,z,x_err,y_err,z_err])
pos, prob, state = sampler.run_mcmc(p0, 200)
sampler.reset()
sampler.run_mcmc(pos, 2000)
fig = p.figure(1,figsize=(6.5,9.5))


# p.subplot(321)
# p.plot(sampler.flatchain[:,0],color='gray')
# p.ylabel(r'$\theta$')
# p.xlabel(r'MCMC Step')

subp = fig.add_subplot(321)
p.hist(np.tan(sampler.flatchain[:,1]),bins=50,color='gray')
p.xlabel(r'$R_{21}$')
p.text(0.05,0.90,'(a)',transform=subp.transAxes)


subp=fig.add_subplot(322)
p.hist(1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1]),\
           bins=50,color='gray')
p.xlabel(r'$R_{32}$')
p.text(0.05,0.90,'(b)',transform=subp.transAxes)


subp=fig.add_subplot(323)

xlike = np.tan(sampler.flatchain[:,1])
ylike = sampler.flatchain[:,3]

values = np.vstack([xlike,ylike])
kernel = scipy.stats.gaussian_kde(values)

xe = np.arange(0.4,0.7,0.005)
ye = np.arange(0.5,2.0,0.02)
xgrid,ygrid = np.meshgrid(xe,ye)
zsurf = kernel([xgrid.ravel(),ygrid.ravel()])
zsurf = zsurf.reshape(xgrid.shape)

p.imshow(zsurf,cmap = 'Greys',aspect='auto',extent = [xe[0],xe[-1],ye[0],ye[-1]],origin='lower')
zvals = np.sort(zsurf.ravel())
cdf = np.cumsum(zvals)
cdf = cdf/cdf.max()
interpolator = interp1d(cdf,zvals)
cvals = interpolator(1-np.array([0.9953,0.9544,0.8427]))
p.contour(xe,ye,zsurf,levels=cvals,colors='k')
p.xlabel(r'$R_{21}$')
p.ylabel(r'$x_{off}$ (K km s$^{-1}$)')
p.text(0.05,0.90,'(c)',transform=subp.transAxes)

subp = fig.add_subplot(324)

xlike = 1/np.tan(sampler.flatchain[:,0])/np.sin(sampler.flatchain[:,1])
ylike = np.sqrt(sampler.flatchain[:,2])

values = np.vstack([xlike,ylike])
kernel = scipy.stats.gaussian_kde(values)

xe = np.arange(0.4,0.7,0.005)
ye = np.arange(0.0,2.0,0.02)
xgrid,ygrid = np.meshgrid(xe,ye)
zsurf = kernel([xgrid.ravel(),ygrid.ravel()])
zsurf = zsurf.reshape(xgrid.shape)

p.imshow(zsurf,cmap = 'Greys',aspect='auto',extent = [xe[0],xe[-1],ye[0],ye[-1]],origin='lower')
zvals = np.sort(zsurf.ravel())
cdf = np.cumsum(zvals)
cdf = cdf/cdf.max()
interpolator = interp1d(cdf,zvals)
cvals = interpolator(1-np.array([0.9953,0.9544,0.8427]))
p.contour(xe,ye,zsurf,levels=cvals,colors='k')
p.xlabel(r'$R_{32}$')
p.ylabel(r'$\sqrt{V}$ (K km s$^{-1}$)')
p.text(0.05,0.90,'(d)',transform=subp.transAxes)

#p.hexbin(np.tan(sampler.flatchain[:,1]),\
#             sampler.flatchain[:,3],cmap='Greys',bins=50)

subp = fig.add_subplot(325)
p.errorbar(x,y,xerr=x_err,yerr=y_err,fmt=None,marker=None,mew=0,\
               color='gray',ecolor='gray',capsize=0)
#p.scatter(x,y,marker='.',color='black')
p.xlabel('CO(1-0)')
p.ylabel('CO(2-1)')
testx = np.linspace(np.nanmin(x),np.nanmax(x),10)
xoff = np.median(sampler.flatchain[:,3])
p.plot(testx,np.tan(np.median(sampler.flatchain[:,1]))*(testx+xoff),color='black')
p.text(0.05,0.90,'(e)',transform=subp.transAxes)

subp = fig.add_subplot(326)
p.errorbar(y,z,xerr=y_err,yerr=z_err,fmt=None,marker=None,mew=0,color='gray',ecolor='gray',capsize=0)
#p.scatter(y,z,marker='.',color='black')
p.xlabel('CO(2-1)')
p.ylabel('CO(3-2)')


testx = np.linspace(np.nanmin(y),np.nanmax(y),10)
p.plot(testx,testx/np.tan(np.median(sampler.flatchain[:,0]))/\
           np.sin(np.median(sampler.flatchain[:,1])),color='black')

p.text(0.05,0.90,'(f)',transform=subp.transAxes)

fig.savefig('sample_figure.pdf',format='pdf')

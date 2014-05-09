import numpy as np
import scipy.stats
import astropy.io.fits as fits
import emcee
import matplotlib.pyplot as p
from astropy.table import Table, Column
from matplotlib import rc
import Radex

temps = np.linspace(5,100,num=30)
Nco = 1e1**(np.linspace(14,17,num=30))
density = 1e1**(np.linspace(2,5,num=30))
t = Table(names=('Temperature','Column','Density','R21','R32','R31'))




ctr = 0
for T in temps:
    for column in Nco: 
        for n in density:
            t.add_row()
            Lines = Radex.RunRadex(KineticTemperature = T, NumberDensity = n,\
                                       ColumnDensity = column, Molecule = 'co',\
                                       UpperFrequencyGHz = 350, LineWidth = 8.0)
            t['R21'][ctr] = Lines[1]['Flux']/Lines[0]['Flux']
            t['R32'][ctr] = Lines[2]['Flux']/Lines[1]['Flux']
            t['R31'][ctr] = Lines[2]['Flux']/Lines[0]['Flux']
            t['Temperature'][ctr] = T
            t['Column'][ctr] = column
            t['Density'][ctr] = n
            ctr = ctr+1


# RadexLogProb(p, co10 = co10,co21 = co21,co32 = co32, 
#                  thirteenco21 = thirteenco21,
#                  co10_err = co10_err, co21_err = co21_err,
#                  co32_err = co32_err,
#                  thirteenco21_err = thirteenco21_err,
#                  linewidth = linewidth):
#     Lines12CO = Radex.RunRadex(KineticTemperature = p[0], NumberDensity = 1e1**p[1],
#                                ColumnDensity = 1e1**p[2], Molecule = 'co',
#                                UpperFrequencyGHz = 350, LineWidth = linewidth)
# # shutting off depletion
#     Lines13CO = Radex.RunRadex(KineticTemperature = p[0], NumberDensity = 1e1**p[1],
#                                ColumnDensity = 1e1**p[2]*0.02, Molecule = '13co',
#                                UpperFrequencyGHz = 230, LowerFrequencyGHz = 210,
#                                LineWidth = linewidth)
#     try:
#         lp = -0.5*(co10 - p[3]*Lines12CO[0]['Flux'])**2/(co10_err**2)+\
#             -0.5*(co21 - p[3]*Lines12CO[1]['Flux'])**2/(co21_err**2)+\
#             -0.5*(co32 - p[3]*Lines12CO[2]['Flux'])**2/(co32_err**2)+\
#             -0.5*(thirteenco21 - p[3]*Lines13CO[0]['Flux'])**2/(thirteenco21_err**2)+\
#             s.beta.logpdf(p[3],1,1)+s.norm.logpdf(p[1],3,1)+s.norm.logpdf(p[2],17,2)+\
#             s.invgamma.logpdf(p[0]/10,0.2,0)
# #            s.beta.logpdf((p[4]-0.018)/0.04,1,1)+
#     except TypeError:
#         lp = -np.inf
# #    print(lp)
#     return lp



# print(RadexLogProb([10,4,17,0.2,1/50.], co10 = co10,co21 = co21,co32 = co32,
#                  thirteenco21 = thirteenco21,
#                  co10_err = co10_err, co21_err = co21_err,
#                  co32_err = co32_err,
#                  thirteenco21_err = thirteenco21_err,
#                  linewidth = linewidth))

# ndim, nwalkers = 5,100
# p0 = (np.random.rand(ndim*nwalkers).reshape((nwalkers,ndim))*2-1)*0.1+1
# xnull,ynull = np.meshgrid([20,3,17,0.2,1/50.],np.ones(nwalkers))
# p0 = p0*xnull
# #p0 = np.array([10,4,17,0.2,1/50.])#*(1+(2*np.random.rand(ndim)-1)*0.2)
# #print(p0)
# sampler = emcee.EnsembleSampler(nwalkers, ndim, RadexLogProb, threads=10)#, args=[co10,co21,co32,thirteenco21,co10_err,co21_err,co32_err,thirteenco21_err,linewidth])
# sampler.run_mcmc(p0, 1000)

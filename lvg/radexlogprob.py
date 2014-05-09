#Define a fixed vector of parameters for radex
# p[0] = Tkin
# p[1] = log10(NumDens)
# p[2] = log10(ColDens)
# p[3] = fillfrac
# p[4] = Depletion

import scipy.stats as stats
def RadexLogProb(p, co10 = 1.0 ,co21 = 1.0 ,co32 = 1.0, \
                     thirteenco = 0.02,\
                     co10_err = 0.1, co21_err = 0.2,\
                     co32_err = 0.1,\
                     thirteenco_err = 0.1,\
                     linewidth = 1.0):
    
    Lines12CO = Radex.RunRadex(KineticTemperature = p[0], NumberDensity = 1e1**p[1],
                               ColumnDensity = 1e1**p[2], Molecule = 'co',
                               UpperFrequencyGHz = 350, LineWidth = linewidth)
    # # shutting off depletion
    Lines13CO = Radex.RunRadex(KineticTemperature = p[0], NumberDensity = 1e1**p[1],
                               ColumnDensity = 1e1**p[2]*0.02, Molecule = '13co',
                               UpperFrequencyGHz = 100, LowerFrequencyGHz = 112,
                               LineWidth = linewidth)
    try:
        lp = -0.5*(co10 - p[3]*Lines12CO[0]['Flux'])**2/(co10_err**2)+\
            -0.5*(co21 - p[3]*Lines12CO[1]['Flux'])**2/(co21_err**2)+\
            -0.5*(co32 - p[3]*Lines12CO[2]['Flux'])**2/(co32_err**2)+\
            -0.5*(thirteenco - p[3]*Lines13CO[0]['Flux'])**2/(thirteenco_err**2)+\
            stats.beta.logpdf(p[3],1,1)+stats.norm.logpdf(p[1],3,1.5)
#+\
#            stats.norm.logpdf(p[2],17,2)+\
#            stats.invgamma.logpdf(p[0]/10,0.2,0)
    #            stats.beta.logpdf((p[4]-0.018)/0.04,1,1)+
    except TypeError:
        lp = -np.inf
    return lp

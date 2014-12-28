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
try:
    import mpi4py
    import emcee.utils import MPIPool
    haveMPI = True
except:
    haveMPI = False
    pass


def logprob3d_checkbaddata(sampler,x,y,z,x_err,y_err,z_err):
    theta,phi,scatter,badfrac,badsig,badmn = sampler.flatchain[:,0],\
        sampler.flatchain[:,1],\
        sampler.flatchain[:,2],\
        sampler.flatchain[:,3],\
        sampler.flatchain[:,4],\
        sampler.flatchain[:,5]
    xoff=0
    pbad = np.zeros(x.size)
    for idx,value in enumerate(x):
        Gamma = (x[idx]+xoff)*np.sin(theta)*np.cos(phi)+\
            y[idx]*np.sin(theta)*np.sin(phi)+z[idx]*np.cos(theta)
        DeltaX2 = ((x[idx]+xoff)-Gamma*np.sin(theta)*np.cos(phi))**2
        DeltaY2 = (y[idx]-Gamma*np.sin(theta)*np.sin(phi))**2
        DeltaZ2 = (z[idx]-Gamma*np.cos(theta))**2
        Delta2 = DeltaX2+DeltaY2+DeltaZ2
        Sigma2 = DeltaX2/Delta2*(x_err[idx]**2+scatter**2)+\
            DeltaY2/Delta2*(y_err[idx]**2+scatter**2)+\
            DeltaZ2/Delta2*(z_err[idx]**2+scatter**2)
        goodlp = -0.5*(Delta2/Sigma2)
        BadDelta = (x[idx]-badmn*np.cos(phi)*np.sin(theta))**2+\
            (y[idx]-badmn*np.sin(phi)*np.sin(theta))**2+\
            (z[idx]-badmn*np.cos(theta))**2
        badlp =-0.5*(BadDelta/(Sigma2+badsig**2))
# run percentiles over chains!
        pbad[idx] = np.percentile(np.exp(badlp)/(np.exp(badlp)+np.exp(goodlp)),50)
    return pbad



def logprob3d_xoff_checkbaddata(sampler,x,y,z,x_err,y_err,z_err):
    theta,phi,xoff,scatter,badfrac,badsig,badmn = sampler.flatchain[:,0],\
        sampler.flatchain[:,1],\
        sampler.flatchain[:,2],\
        sampler.flatchain[:,3],\
        sampler.flatchain[:,4],\
        sampler.flatchain[:,5],\
        sampler.flatchain[:,6]
    pbad = np.zeros(x.size)
    for idx,value in enumerate(x):
        Gamma = (x[idx]+xoff)*np.sin(theta)*np.cos(phi)+\
            y[idx]*np.sin(theta)*np.sin(phi)+z[idx]*np.cos(theta)
        DeltaX2 = ((x[idx]+xoff)-Gamma*np.sin(theta)*np.cos(phi))**2
        DeltaY2 = (y[idx]-Gamma*np.sin(theta)*np.sin(phi))**2
        DeltaZ2 = (z[idx]-Gamma*np.cos(theta))**2
        Delta2 = DeltaX2+DeltaY2+DeltaZ2
        Sigma2 = DeltaX2/Delta2*(x_err[idx]**2+scatter**2)+\
            DeltaY2/Delta2*(y_err[idx]**2+scatter**2)+\
            DeltaZ2/Delta2*(z_err[idx]**2+scatter**2)
        goodlp = -0.5*(Delta2/Sigma2)
        BadDelta = (x[idx]-badmn*np.cos(phi)*np.sin(theta))**2+\
            (y[idx]-badmn*np.sin(phi)*np.sin(theta))**2+\
            (z[idx]-badmn*np.cos(theta))**2
        badlp =-0.5*(BadDelta/(Sigma2+badsig**2))
# run percentiles over chains!
        pbad[idx] = np.percentile(np.exp(badlp)/(np.exp(badlp)+np.exp(goodlp)),50)
    return pbad

def logprob2d_checkbaddata(sampler,x,y,x_err,y_err):
    if sampler.flatchain.shape[1]==7:
        theta,xoff,scatter,badfrac,xbad,ybad,badsig = sampler.flatchain[:,0],\
                                                  sampler.flatchain[:,1],\
                                                  sampler.flatchain[:,2],\
                                                  sampler.flatchain[:,3],\
                                                  sampler.flatchain[:,4],\
                                                  sampler.flatchain[:,5],\
                                                  sampler.flatchain[:,6]

    if sampler.flatchain.shape[1]==6:
        theta,scatter,badfrac,xbad,ybad,badsig = \
            sampler.flatchain[:,0],\
            sampler.flatchain[:,1],\
            sampler.flatchain[:,2],\
            sampler.flatchain[:,3],\
            sampler.flatchain[:,4],\
            sampler.flatchain[:,5]
        xoff=0
    if sampler.flatchain.shape[1] == 5:
        theta,scatter,badfrac,badsig,badmn = sampler.flatchain[:,0],\
                                             sampler.flatchain[:,1],\
                                             sampler.flatchain[:,2],\
                                             sampler.flatchain[:,3],\
                                             sampler.flatchain[:,4]
        xoff=0
    pbad = np.zeros(x.size)
    for idx,value in enumerate(x):
        Delta = (np.cos(theta)*y[idx] - np.sin(theta)*(x[idx]+xoff))**2
        Sigma = (np.sin(theta))**2*(x_err[idx]**2+scatter**2)+\
            (np.cos(theta))**2*(y_err[idx]**2+scatter**2)
        goodlp = -0.5*(Delta/Sigma)
        BadDelta = (y[idx]-ybad)**2+(x[idx]-xbad)**2
        badlp =-0.5*(BadDelta/(Sigma+badsig**2))
# run percentiles over chains!
        pbad[idx] = np.percentile(np.exp(badlp)/(np.exp(badlp)+np.exp(goodlp)),50)
    return pbad


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

def summarize2d(t,sampler21=None,sampler32=None):
    """
    Summarizes a sampler behavior into the output table

    """

    if sampler21 is not None:
        r21 = np.tan(sampler21.flatchain[:,0])
        t['R21'][-1] = np.median(r21)
        t['R21+'][-1] = scipy.stats.scoreatpercentile(r21,85)-t['R21'][-1]
        t['R21-'][-1] = t['R21'][-1]-scipy.stats.scoreatpercentile(r21,15)

    if sampler32 is not None:
        r32 = np.tan(sampler32.flatchain[:,0])
        t['R32'][-1] = np.median(r32)
        t['R32+'][-1] = scipy.stats.scoreatpercentile(r32,85)-t['R32'][-1]
        t['R32-'][-1] = t['R32'][-1]-scipy.stats.scoreatpercentile(r32,15)
        
def table_template():
    t = Table(names=('Name','theta','theta+','theta-','phi','phi+','phi-',\
                     'R21-','R21','R21+','R32-','R32','R32+',
                     'xR21-','xR21','xR21+','xR32-','xR32','xR32+',
                     'var32','var21',
                     'R31-','R31','R31+','var','xoff','LowKey',\
                     'HighKey','MedKey',\
                     'LowKey21','MedKey21','HighKey21',\
                     'LowKey32','MedKey32','HighKey32',\
                     'Npts','Npts21','Npts32'),\
              dtype=('S7','f8','f8','f8','f8','f8','f8','f8','f8','f8',\
                          'f8','f8','f8','f8','f8','f8','f8','f8',\
                          'f8','f8','f8','f8','f8','f8','f8','f8',\
                          'f8','f8','f8','f8','f8','f8',\
                          'f8','f8','f8','f8','f8','f8'))
    return t

def bygal(fitsfile,spire_cut=3.0):
    s = fits.getdata(fitsfile)
    hdr = fits.getheader(fitsfile)
    GalNames = np.unique(s['GALNAME'])
    
    cut = -2

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

        SignifData = ((s['CO10']>cut*s['CO10_ERR'])&
                        (s['CO21']>cut*s['CO21_ERR'])&
                        (s['CO32']>cut*s['CO32_ERR'])&
                        (s['INTERF']==0)&
                        (s['GALNAME']==name)&
                        (s['SPIRE1']> spire_cut))
        sub = s[SignifData]

        idx21 = np.where((s['GALNAME']==name)&
                         (s['CO10']>cut*s['CO10_ERR'])&
                         (s['CO21']>cut*s['CO21_ERR'])&
                         (s['SPIRE1']> spire_cut))
        sub21 = s[idx21]

        idx32 = np.where((s['GALNAME']==name)&
                         (s['CO32']>cut*s['CO32_ERR'])&
                         (s['CO21']>cut*s['CO21_ERR'])&
                         (s['SPIRE1']>spire_cut))

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
            # ndim, nwalkers = 5,50
            # p0 = np.zeros((nwalkers,ndim))
            # p0[:,0] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
            # p0[:,1] = np.pi/4+np.random.randn(nwalkers)*np.pi/8
            # p0[:,2] = (np.random.randn(nwalkers))**2
            # p0[:,3] = (np.random.randn(nwalkers))**2
            # p0[:,4] = (np.random.randn(nwalkers))**2

            ndim, nwalkers = 7,50
            p0 = np.zeros((nwalkers,ndim))
            p0[:,0] = np.pi/6+np.random.randn(nwalkers)*np.pi/8
            p0[:,1] = np.pi/6+np.random.randn(nwalkers)*np.pi/8
            p0[:,2] = (np.random.randn(nwalkers))*np.median(x_err) # xoffset
            p0[:,3] = (np.random.randn(nwalkers))**2*(np.median(x_err)**2+np.median(y_err)**2+np.median(z_err)**2) # scatter
            p0[:,4] = (np.random.randn(nwalkers)*0.01)**2 # bad fraction
            p0[:,5] = np.percentile(x,95)+np.random.randn(nwalkers)*np.median(x_err)
            p0[:,6] = np.percentile(x,90)+np.median(x_err)*np.random.randn(nwalkers)
            
            
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lp.logprob3d_xoff_scatter_mixture,
                                        args=[x,y,z,x_err,y_err,z_err])
            pos, prob, state = sampler.run_mcmc(p0, 400)
            sampler.reset()
            sampler.run_mcmc(pos,1000)
            print(name,np.mean(sampler.acceptance_fraction),
                  1/np.tan(np.median(sampler.flatchain[:,0])))
            badprob = logprob3d_xoff_checkbaddata(sampler,x,y,z,x_err,y_err,z_err)
            splt.sampler_plot_mixture(sampler,data,name=name,badprob=badprob)
            summarize(t,sampler)
            t.write('brs.bygal.txt',format='ascii')
        it.iternext()

def bycategory(fitsfile,category=['TDEP','RGAL','SPIRE1','RGALNORM','FUV',
                                  'UVCOLOR','SFR','IRCOLOR',
                                  'STELLARSD','MOLRAT','PRESSURE','RGAL'],
                                  spire_cut=3.0, withMPI = False):
    if withMPI:
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        else:
            pool = None

    category = np.array(category)
    s = fits.getdata(fitsfile)
    hdr = fits.getheader(fitsfile)
    GalNames = np.unique(s['GALNAME'])

    cut = -2
    quantile = 10
    nValid = 1

    dq = 1e2/quantile
    lower_percentiles = np.arange(quantile)*dq
    upper_percentiles = np.arange(quantile)*dq+dq

    it = np.nditer(lower_percentiles,flags=['f_index'])
    t=table_template()
    nGal = len(GalNames)

    # Identify significant emission on keys

    SignifData = ((s['CO10']>cut*s['CO10_ERR'])&
                  (s['CO21']>cut*s['CO21_ERR'])&
                  (s['CO32']>cut*s['CO32_ERR'])&
                  (s['INTERF']==0)&
                  (s['SPIRE1']> spire_cut))
    sub = s[SignifData]

    
    

    Signif21 = ((s['CO10']>cut*s['CO10_ERR'])&\
                (s['CO21']>cut*s['CO21_ERR'])&\
                (s['INTERF']==0)&\
                ((s['SPIRE1']> spire_cut)|(np.isnan(s['SPIRE1']))))

    Signif32 = ((s['CO32']>cut*s['CO32_ERR'])&\
                (s['CO21']>cut*s['CO21_ERR'])&\
                ((s['SPIRE1']> spire_cut)|(np.isnan(s['SPIRE1']))))
    
    sub21 = s[Signif21]
    sub32 = s[Signif32]
    
    molrat = 313*s['CO21']/s['HI']
    sfr = 634*s['HA']+0.00325*s['MIPS24']
    ircolor = (s['MIPS24']/s['PACS3'])
    stellarsd = 200*s['IRAC1']
    pressure = 272*(s['HI']*0.02+s['CO21']*6.7)*\
               np.sqrt(stellarsd)*8/np.sqrt(212)
    tdep = 6.7*s['CO21']/sfr*1e6


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
        elif keyname == 'TDEP':
            key_variable = tdep
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
            name = (keyname+'_'+np.array_str(it.value)).upper()
            t.add_row()
            t['Name'][-1] = name

            lower_score =scipy.stats.scoreatpercentile(keyscores,\
                                                           lower_percentiles[pct])
            upper_score =scipy.stats.scoreatpercentile(keyscores,\
                                                           upper_percentiles[pct])

            t['LowKey'][-1] = np.log10(lower_score)
            t['HighKey'][-1] = np.log10(upper_score)
            print(lower_score,upper_score)
            idx = np.where((key_variable>=lower_score)&
                           (key_variable<=upper_score)&(SignifData))
            sub = s[idx]
            t['MedKey'][-1] =np.log10(np.median(key_variable[idx]))

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
                t['Npts'][-1]=x.size
                data = dict(x=x,x_err=x_err,y=y,y_err=y_err,z=z,z_err=z_err)

                ndim, nwalkers = 6,50
                p0 = np.zeros((nwalkers,ndim))
                p0[:,0] = np.pi/6+np.random.randn(nwalkers)*np.pi/8
                p0[:,1] = np.pi/6+np.random.randn(nwalkers)*np.pi/8
                p0[:,2] = (np.random.randn(nwalkers))**2*(np.median(x_err)**2+np.median(y_err)**2+np.median(z_err)**2) # scatter
                p0[:,3] = (np.random.randn(nwalkers)*0.01)**2 # bad fraction
                p0[:,4] = np.percentile(x,95)+np.random.randn(nwalkers)*np.median(x_err)
                p0[:,5] = np.percentile(x,90)+np.median(x_err)*np.random.randn(nwalkers)

                sampler = emcee.EnsembleSampler(nwalkers, ndim, lp.logprob3d_scatter_mixture,
                                            args=[x,y,z,x_err,y_err,z_err], pool = pool)

                pos, prob, state = sampler.run_mcmc(p0, 400)
                sampler.reset()
                sampler.run_mcmc(pos,1000)
                print(name,np.mean(sampler.acceptance_fraction),
                      1/np.tan(np.median(sampler.flatchain[:,0])))
                badprob = logprob3d_checkbaddata(sampler,x,y,z,x_err,y_err,z_err)
                splt.sampler_plot_mixture(sampler,data,name=name,badprob=badprob)
                summarize(t,sampler)

            it.iternext()
        t.write('brs_category.'+keyname+'.txt',format='ascii')
        t2 = Table(t,copy=True)
        t2.remove_columns(('theta','phi','theta+','phi+','theta-','phi-'))
        emptystring = np.empty((len(t2)),dtype='string')
        emptystring[:]=''
        col = Column(name='blank',data=emptystring)
        t2.add_column(col,index=4)
        col = Column(name='blank2',data=emptystring)
        t2.add_column(col,index=8)
        t2.write('brs_category.'+keyname+'.tex',format='latex')
    iter2.iternext()


def bygal2d(fitsfile,spire_cut=3.0):
    s = fits.getdata(fitsfile)
    hdr = fits.getheader(fitsfile)
    GalNames = np.unique(s['GALNAME'])
    
    cut = -2

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

        idx21 = np.where((s['GALNAME']==name)&
                         (s['CO10']>cut*s['CO10_ERR'])&
                         (s['CO21']>cut*s['CO21_ERR'])&
                          ((s['SPIRE1']> spire_cut)|(np.isnan(s['SPIRE1']))))

        sub21 = s[idx21]

        idx32 = np.where((s['GALNAME']==name)&
                         (s['CO32']>cut*s['CO32_ERR'])&
                         (s['CO21']>cut*s['CO21_ERR'])&
                         ((s['SPIRE1']> spire_cut)|(np.isnan(s['SPIRE1']))))


        sub32 = s[idx32]

        print('Number of r21 points: {0}. Number of r32 points: {1}'.format(len(sub21),len(sub32)))
        if len(sub21)>1:
            x = sub21['CO10']
            x_err = sub21['CO10_ERR']
            y = sub21['CO21']
            y_err = sub21['CO21_ERR']

            t['Npts'][-1]=x.size
            data = dict(x=x,x_err=x_err,y=y,y_err=y_err)
            ndim, nwalkers = 7,50
            p0 = np.zeros((nwalkers,ndim))
            p0[:,0] = np.pi/6+np.random.randn(nwalkers)*np.pi/8
            p0[:,1] = np.percentile(y,95)+np.random.randn(nwalkers)*np.median(x_err)
            p0[:,2] = (np.random.randn(nwalkers))**2*(np.median(x_err)**2+np.median(y_err)**2) # scatter
            p0[:,3] = (np.random.randn(nwalkers)*0.01)**2 # bad fraction
            p0[:,4] = np.percentile(x,95)+np.random.randn(nwalkers)*np.median(x_err)
            p0[:,5] = np.percentile(y,95)+np.random.randn(nwalkers)*np.median(x_err)
            p0[:,6] = np.percentile(x,90)+np.median(x_err)*np.random.randn(nwalkers)
            
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lp.logprob2d_xoff_scatter_mixture,
                                        args=[x,y,x_err,y_err])
            pos, prob, state = sampler.run_mcmc(p0, 400)
            sampler.reset()
            sampler.run_mcmc(pos,1000)
            print('Name {0}, Acceptance Fraction {1}, Ratio {2}'.format(name,np.mean(sampler.acceptance_fraction),
                                                                                     np.tan(np.median(sampler.flatchain[:,0]))))
            badprob = logprob2d_checkbaddata(sampler,x,y,x_err,y_err)
            splt.sampler_plot2d_mixture(sampler,data,name=name+'.21',badprob=badprob)
            summarize2d(t,sampler21=sampler)

        if len(sub32)>1:
            x = sub32['CO21']
            x_err = sub32['CO21_ERR']
            y = sub32['CO32']
            y_err = sub32['CO32_ERR']
            t['Npts'][-1]=x.size
            data = dict(x=x,x_err=x_err,y=y,y_err=y_err)

            ndim, nwalkers = 6,50
            p0 = np.zeros((nwalkers,ndim))
            p0[:,0] = np.pi/6+np.random.randn(nwalkers)*np.pi/8
            p0[:,1] = (np.random.randn(nwalkers))**2*(np.median(x_err)**2+np.median(y_err)**2) # scatter
            p0[:,2] = (np.random.randn(nwalkers)*0.01)**2 # bad fraction
            p0[:,3] = np.percentile(x,95)+np.random.randn(nwalkers)*np.median(x_err)
            p0[:,4] = np.percentile(y,95)+np.random.randn(nwalkers)*np.median(x_err)
            p0[:,5] = np.percentile(x,90)+np.median(x_err)*np.random.randn(nwalkers)

            sampler = emcee.EnsembleSampler(nwalkers, ndim, lp.logprob2d_scatter_mixture,
                                        args=[x,y,x_err,y_err])
            pos, prob, state = sampler.run_mcmc(p0, 400)
            sampler.reset()
            sampler.run_mcmc(pos,1000)
            print('Name {0}, Acceptance Fraction {1}, Ratio {2}'.format(name,np.mean(sampler.acceptance_fraction),
                                                                                     np.tan(np.median(sampler.flatchain[:,0]))))
            badprob = logprob2d_checkbaddata(sampler,x,y,x_err,y_err)
            splt.sampler_plot2d_mixture(sampler,data,name=name+'.32',badprob=badprob)
            summarize2d(t,sampler32=sampler)
            t.write('brs.bygal2d.txt',format='ascii')
        it.iternext()

def bycategory2d(fitsfile,category=['TDEP','RGAL','SPIRE1','RGALNORM','FUV',
                                  'UVCOLOR','SFR','IRCOLOR',
                                  'STELLARSD','MOLRAT','PRESSURE'],
                                  spire_cut=3.0):
    category = np.array(category)
    s = fits.getdata(fitsfile)
    hdr = fits.getheader(fitsfile)
    GalNames = np.unique(s['GALNAME'])

    cut = -2
    quantile = 10
    nValid = 1

    dq = 1e2/quantile
    lower_percentiles = np.arange(quantile)*dq
    upper_percentiles = np.arange(quantile)*dq+dq

    it = np.nditer(lower_percentiles,flags=['f_index'])
    t=table_template()
    nGal = len(GalNames)

    # Identify significant emission on keys


    Signif21 = ((s['CO10']>cut*s['CO10_ERR'])&\
                (s['CO21']>cut*s['CO21_ERR'])&\
                (s['INTERF']==0)&\
                ((s['SPIRE1']> spire_cut)|(np.isnan(s['SPIRE1']))))

    Signif32 = ((s['CO32']>cut*s['CO32_ERR'])&\
                (s['CO21']>cut*s['CO21_ERR'])&\
                ((s['SPIRE1']> spire_cut)|(np.isnan(s['SPIRE1']))))

    
    molrat = 313*s['CO21']/s['HI']
    sfr = 634*s['HA']+0.00325*s['MIPS24']
    ircolor = (s['MIPS24']/s['PACS3'])
    stellarsd = 200*s['IRAC1']
    pressure = 272*(s['HI']*0.02+s['CO21']*6.7)*\
               np.sqrt(stellarsd)*8/np.sqrt(212)
    tdep = 6.7*s['CO21']/sfr*1e6

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
        elif keyname == 'TDEP':
            key_variable = tdep
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
        keyscores21 = key_variable[np.isfinite(key_variable)&(Signif21)]
        keyscores32 = key_variable[np.isfinite(key_variable)&(Signif32)]
        t=table_template()
        while not it.finished:
            pct = it.index
            name = np.array_str(it.value)
            t.add_row()
            t['Name'][pct] = name.upper()

            lower_score21 =scipy.stats.scoreatpercentile(keyscores21,\
                                                           lower_percentiles[pct])
            upper_score21 =scipy.stats.scoreatpercentile(keyscores21,\
                                                           upper_percentiles[pct])

            lower_score32 =scipy.stats.scoreatpercentile(keyscores32,\
                                                           lower_percentiles[pct])
            upper_score32 =scipy.stats.scoreatpercentile(keyscores32,\
                                                           upper_percentiles[pct])

            t['LowKey21'][pct] = np.log10(lower_score21)
            t['HighKey21'][pct] = np.log10(upper_score21)
            print('Key: {0}. Lower: {1}  Upper: {2}'.format(keyname,lower_score21,upper_score21))
            t['MedKey21'][pct] =np.log10(np.median(keyscores21[(keyscores21>=lower_score21)&
                                                           (keyscores21<=upper_score21)]))

            t['LowKey32'][pct] = np.log10(lower_score32)
            t['HighKey32'][pct] = np.log10(upper_score32)
            print('Key: {0}. Lower: {1}  Upper: {2}'.format(keyname,lower_score32,upper_score32))
            t['MedKey32'][pct] =np.log10(np.median(keyscores32[(keyscores32>=lower_score32)&
                                                           (keyscores32<=upper_score32)]))

            idx21 = np.where((key_variable>=lower_score21)&
                           (key_variable<=upper_score21)&(Signif21))
            idx32 = np.where((key_variable>=lower_score32)&
                            (key_variable<=upper_score32)&(Signif32))

            sub21 = s[idx21]
            sub32 = s[idx32]
            print('Number of r21 points: {0}. Number of r32 points: {1}'.format(len(sub21),len(sub32)))
            if len(sub21)>nValid:
                x = sub21['CO10']
                x_err = sub21['CO10_ERR']
                y = sub21['CO21']
                y_err = sub21['CO21_ERR']
                data = dict(x=x,x_err=x_err,y=y,y_err=y_err)

                ndim, nwalkers = 6,50
                p0 = np.zeros((nwalkers,ndim))
                p0[:,0] = np.pi/6+np.random.randn(nwalkers)*np.pi/8
                p0[:,1] = (np.random.randn(nwalkers))**2*(np.median(x_err)**2+np.median(y_err)**2) # scatter
                p0[:,2] = (np.random.randn(nwalkers)*0.01)**2 # bad fraction
                p0[:,3] = np.percentile(x,95)+np.random.randn(nwalkers)*np.median(x_err)
                p0[:,4] = np.percentile(y,95)+np.random.randn(nwalkers)*np.median(x_err)
                p0[:,5] = np.percentile(x,90)+np.median(x_err)*np.random.randn(nwalkers)

                sampler = emcee.EnsembleSampler(nwalkers, ndim, lp.logprob2d_scatter_mixture,
                                            args=[x,y,x_err,y_err])
                pos, prob, state = sampler.run_mcmc(p0, 400)
                sampler.reset()
                sampler.run_mcmc(pos,1000)
                print('Name {0}, Acceptance Fraction {1}, Ratio {2}'.format(name,np.mean(sampler.acceptance_fraction),
                                                                                         np.tan(np.median(sampler.flatchain[:,0]))))
                badprob = logprob2d_checkbaddata(sampler,x,y,x_err,y_err)
#                pdb.set_trace()
                splt.sampler_plot2d_mixture(sampler,data,name=keyname+'.'+name+'.21',badprob=badprob)
                summarize2d(t,sampler21=sampler)

            if len(sub32)>nValid:
                x = sub32['CO21']
                x_err = sub32['CO21_ERR']
                y = sub32['CO32']
                y_err = sub32['CO32_ERR']
                data = dict(x=x,x_err=x_err,y=y,y_err=y_err)
                ndim, nwalkers = 6,50
                p0 = np.zeros((nwalkers,ndim))
                p0[:,0] = np.pi/6+np.random.randn(nwalkers)*np.pi/8
                p0[:,1] = (np.random.randn(nwalkers))**2*(np.median(x_err)**2+np.median(y_err)**2) # scatter
                p0[:,2] = (np.random.randn(nwalkers)*0.01)**2 # bad fraction
                p0[:,3] = np.percentile(x,95)+np.random.randn(nwalkers)*np.median(x_err)
                p0[:,4] = np.percentile(y,95)+np.random.randn(nwalkers)*np.median(x_err)
                p0[:,5] = np.percentile(x,90)+np.median(x_err)*np.random.randn(nwalkers)

                sampler = emcee.EnsembleSampler(nwalkers, ndim, lp.logprob2d_scatter_mixture,
                                            args=[x,y,x_err,y_err])
                pos, prob, state = sampler.run_mcmc(p0, 400)
                sampler.reset()
                sampler.run_mcmc(pos,1000)
                print('Name {0}, Acceptance Fraction {1}, Ratio {2}'.format(name,np.mean(sampler.acceptance_fraction),
                                                                                         np.tan(np.median(sampler.flatchain[:,0]))))
                badprob = logprob2d_checkbaddata(sampler,x,y,x_err,y_err)
                splt.sampler_plot2d_mixture(sampler,data,name=keyname+'.'+name+'.32',badprob=badprob)
                summarize2d(t,sampler32=sampler)
                t.write('brs_category.'+keyname+'.txt',format='ascii')
            it.iternext()
        iter2.iternext()

def alldata2d(fitsfile):
    s = fits.getdata(fitsfile)
    hdr = fits.getheader(fitsfile)
    GalNames = np.unique(s['GALNAME'])
    
    cut = -4
    spire_cut=3
    nValid = 3
    t = table_template()
    for tag in t.keys():
        if tag != 'Name':
            t[tag].format = '{:.3f}'
    t.add_row()

    idx21 = np.where((s['CO10']>cut*s['CO10_ERR'])&
                     (s['CO21']>cut*s['CO21_ERR'])&
                     (s['SPIRE1']> spire_cut))
    sub21 = s[idx21]

    idx32 = np.where((s['CO32']>cut*s['CO32_ERR'])&
                     (s['CO21']>cut*s['CO21_ERR'])&
                     (s['SPIRE1']>spire_cut))
    sub32 = s[idx32]

    idx31 = np.where((s['CO32']>cut*s['CO32_ERR'])&
                     (s['CO10']>cut*s['CO10_ERR'])&
                     (s['SPIRE1']>spire_cut))
    sub31 = s[idx31]

    print('Number of r21 points: {0}. Number of r32 points: {1}'.format(len(sub21),len(sub32)))
    keyname = 'ALLDATA'
    name = ''
    if len(sub21)>nValid:
        x = sub21['CO10']
        x_err = sub21['CO10_ERR']
        y = sub21['CO21']
        y_err = sub21['CO21_ERR']
        data = dict(x=x,x_err=x_err,y=y,y_err=y_err)

        ndim, nwalkers = 6,50
        p0 = np.zeros((nwalkers,ndim))
        p0[:,0] = np.pi/6+np.random.randn(nwalkers)*np.pi/8
        p0[:,1] = (np.random.randn(nwalkers))**2*(np.median(x_err)**2+np.median(y_err)**2) # scatter
        p0[:,2] = (np.random.randn(nwalkers)*0.01)**2 # bad fraction
        p0[:,3] = np.percentile(x,95)+np.random.randn(nwalkers)*np.median(x_err)
        p0[:,4] = np.percentile(y,95)+np.random.randn(nwalkers)*np.median(x_err)
        p0[:,5] = np.percentile(x,90)+np.median(x_err)*np.random.randn(nwalkers)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp.logprob2d_scatter_mixture,
                                    args=[x,y,x_err,y_err])
        pos, prob, state = sampler.run_mcmc(p0, 400)
        sampler.reset()
        sampler.run_mcmc(pos,1000)
        print('Name {0}, Acceptance Fraction {1}, Ratio {2}'.format(name,np.mean(sampler.acceptance_fraction),
                                                                                 np.tan(np.median(sampler.flatchain[:,0]))))
        badprob = logprob2d_checkbaddata(sampler,x,y,x_err,y_err)
#        pdb.set_trace()
        splt.sampler_plot2d_mixture(sampler,data,name='alldata.21',badprob=badprob)
        summarize2d(t,sampler21=sampler)

    if len(sub32)>nValid:
        x = sub32['CO21']
        x_err = sub32['CO21_ERR']
        y = sub32['CO32']
        y_err = sub32['CO32_ERR']
        data = dict(x=x,x_err=x_err,y=y,y_err=y_err)
        ndim, nwalkers = 6,50
        p0 = np.zeros((nwalkers,ndim))
        p0[:,0] = np.pi/6+np.random.randn(nwalkers)*np.pi/8
        p0[:,1] = (np.random.randn(nwalkers))**2*(np.median(x_err)**2+np.median(y_err)**2) # scatter
        p0[:,2] = (np.random.randn(nwalkers)*0.01)**2 # bad fraction
        p0[:,3] = np.percentile(x,95)+np.random.randn(nwalkers)*np.median(x_err)
        p0[:,4] = np.percentile(y,95)+np.random.randn(nwalkers)*np.median(x_err)
        p0[:,5] = np.percentile(x,90)+np.median(x_err)*np.random.randn(nwalkers)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp.logprob2d_scatter_mixture,
                                    args=[x,y,x_err,y_err])
        pos, prob, state = sampler.run_mcmc(p0, 400)
        sampler.reset()
        sampler.run_mcmc(pos,1000)
        print('Name {0}, Acceptance Fraction {1}, Ratio {2}'.format(name,np.mean(sampler.acceptance_fraction),
                                                                                 np.tan(np.median(sampler.flatchain[:,0]))))
        badprob = logprob2d_checkbaddata(sampler,x,y,x_err,y_err)
        splt.sampler_plot2d_mixture(sampler,data,name='addata.32',badprob=badprob)
        summarize2d(t,sampler32=sampler)
        t.write('alldata.txt',format='ascii')

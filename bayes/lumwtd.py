import astropy.io.fits as fits
import numpy as np

def lumwtd(filename='colira.fits'):
    s = fits.getdata(filename)
    keep = (s['INTERF'] != 2)# & (s['GALNAME']=='ngc3034')
    W10 = np.nansum(s['CO10'][keep])
    W21 = np.nansum(s['CO21'][keep])
    W32 = np.nansum(s['CO32'][keep])
    W10_err = (np.nansum((s['CO10_ERR'][keep])**2))**0.5
    W21_err = (np.nansum((s['CO21_ERR'][keep])**2))**0.5
    W32_err = (np.nansum((s['CO32_ERR'][keep])**2))**0.5

    print('R21 = {0:4f} +/- {1:4f}'.format(\
        W21/W10,W21/W10*((W10_err/W10)**2+(W21_err/W21)**2)**0.5))
    print('R32 = {0:4f} +/- {1:4f}'.format(\
        W32/W21,W32/W21*((W32_err/W32)**2+(W21_err/W21)**2)**0.5))
    print('R31 = {0:4f} +/- {1:4f}'.format(\
        W32/W10,W32/W10*((W32_err/W32)**2+(W10_err/W10)**2)**0.5))

def lumwtd_bygal(filename='colira.fits'):
    s = fits.getdata(filename)
    for name in np.unique(s['GALNAME']):
        keep = (s['INTERF'] == 1) & (s['GALNAME']==name) & (s['SPIRE1']>10)
        if np.any(keep):
            W10 = np.nansum(s['CO10'][keep])
            W21 = np.nansum(s['CO21'][keep])
            W32 = np.nansum(s['CO32'][keep])
            W10_err = (np.nansum((s['CO10_ERR'][keep])**2))**0.5
            W21_err = (np.nansum((s['CO21_ERR'][keep])**2))**0.5
            W32_err = (np.nansum((s['CO32_ERR'][keep])**2))**0.5
            print('Galaxy: '+name)
            print('R21 = {0:4f} +/- {1:4f}'.format(\
                W21/W10,W21/W10*((W10_err/W10)**2+(W21_err/W21)**2)**0.5))
            print('R32 = {0:4f} +/- {1:4f}'.format(\
                W32/W21,W32/W21*((W32_err/W32)**2+(W21_err/W21)**2)**0.5))
            print('R31 = {0:4f} +/- {1:4f}'.format(\
                W32/W10,W32/W10*((W32_err/W32)**2+(W10_err/W10)**2)**0.5))

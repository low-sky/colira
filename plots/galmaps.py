from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy.interpolate import griddata
#rc('text', usetex=True)
#rc('font',family='serif')

def galmaps(fitsfile,plotdir='../plots/'):
    t = Table.read(fitsfile)
    names = t['GALNAME']
    unames = np.unique(names)
    for thisgal in unames:
        idx = np.where(thisgal == names)
        ra0 = np.median(t[idx]['RA'])
        dec0 = np.median(t[idx]['DEC'])
        plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.scatter(t[idx]['RA']-ra0,t[idx]['DEC']-dec0,c=t[idx]['CO10'],marker='h',linewidth=0,cmap='winter')
        cb = plt.colorbar()
        cb.set_label('CO(1-0)')
        plt.subplot(132)
        plt.scatter(t[idx]['RA']-ra0,t[idx]['DEC']-dec0,c=t[idx]['CO21'],marker='h',linewidth=0,cmap='winter')
        cb = plt.colorbar()
        cb.set_label('CO(2-1)')
        plt.subplot(133)
        plt.scatter(t[idx]['RA']-ra0,t[idx]['DEC']-dec0,c=t[idx]['CO32'],marker='h',linewidth=0,cmap='winter')
        cb = plt.colorbar()
        cb.set_label('CO(3-2)')
        plt.tight_layout()
        plt.title(thisgal)
        plt.savefig(plotdir+thisgal+'.maps.pdf')

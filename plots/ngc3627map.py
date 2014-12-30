import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits

def soften(x,percentile=99):
    knee = np.percentile(x,percentile)
    #    xnew = np.arctan(x/knee)/0.78539*knee
    index = np.where(x>knee)
    xnew = np.copy(x)
    xnew[index] = knee
    return(xnew)

def maps(fitsfile):
    s = fits.getdata(fitsfile)
    hdr = fits.getheader(fitsfile)
    spire_cut = 3.0
    cut = 3.0
    SignifData = ((s['CO21']>cut*s['CO21_ERR'])&
                  (np.isfinite(s['CO32']))&
                  (np.isfinite(s['CO10']))&
                  (s['GALNAME']=='ngc3627')&
                  (s['SPIRE1']> spire_cut))    
    sub = s[SignifData]
    ra0 = 170.0623508
    dec0 = 12.9915378

    # NGC 0628
    # ra0 = 24.1739458
    # dec0 = 15.7836619
    
    x = ((sub['RA'] - ra0)*np.cos(dec0*np.pi/180))*3600
    y = (sub['DEC'] - dec0)*3600

    alp = 1
    lsz = 8
    mkr = 'h'
    cm = 'ocean_r'
    ms = 10
    fig = plt.figure(figsize=(9,6))

    ax = plt.subplot(241)
    plt.scatter(x,y,marker=mkr,c=soften(sub['CO10']),edgecolors='none',zorder=100,
                cmap=cm,alpha=alp,s=ms)
    ax.xaxis.set_visible(False)
    ax.set_aspect('equal')
    cb = plt.colorbar(orientation='horizontal',pad=0.05)
    cb.set_label('CO(1-0) [K km/s]')
    cb.ax.tick_params(labelsize=lsz)
    plt.ylabel('DEC (offset, arcsec)')

    ax = plt.subplot(242)
    plt.scatter(x,y,marker=mkr,c=soften(sub['CO21']),edgecolors='none',
                zorder=100,cmap=cm,alpha=alp,s=ms)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_aspect('equal')
    cb = plt.colorbar(orientation='horizontal',pad=0.05)
    cb.set_label('CO(2-1) [K km/s]')
    cb.ax.tick_params(labelsize=lsz)

    ax = plt.subplot(243)
    plt.scatter(x,y,marker=mkr,c=soften(sub['CO32']),edgecolors='none',
                zorder=100,cmap=cm,s=ms)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_aspect('equal')
    cb = plt.colorbar(orientation='horizontal',pad=0.05)
    cb.set_label('CO(3-2) [K km/s]')
    cb.ax.tick_params(labelsize=lsz)

    ax = plt.subplot(244)
    plt.scatter(x,y,marker=mkr,c=soften(sub['MIPS24']),edgecolors='none',
                zorder=100,cmap=cm,s=ms)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_aspect('equal')
    cb = plt.colorbar(orientation='horizontal',pad=0.05)
    cb.set_label(r'24 $\mu$m [MJy/sr]')
    cb.ax.tick_params(labelsize=lsz)


    ax = plt.subplot(245)
    plt.scatter(x,y,marker=mkr,c=soften(sub['HI']),
                edgecolors='none',zorder=100,cmap=cm,s=ms)
    ax.xaxis.set_visible(False)
    ax.set_aspect('equal')
    plt.ylabel('DEC (offset, arcsec)')
    cb = plt.colorbar(orientation='horizontal',pad=0.05)
    cb.set_label(r'HI [K km/s]')
    cb.ax.tick_params(labelsize=lsz)

    ax = plt.subplot(246)
    plt.scatter(x,y,marker=mkr,c=soften(sub['SPIRE1']),edgecolors='none',
                zorder=100,cmap=cm,s=ms)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_aspect('equal')
    cb = plt.colorbar(orientation='horizontal',pad=0.05)
    cb.set_label(r'250 $\mu$m [MJy/sr]')
    cb.ax.tick_params(labelsize=lsz)

    ax = plt.subplot(247)
    plt.scatter(x,y,marker=mkr,c=soften(sub['IRAC1']),
                edgecolors='none',zorder=100,cmap=cm,s=ms)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_aspect('equal')
    cb = plt.colorbar(orientation='horizontal',pad=0.05)
    cb.set_label(r'3.6 $\mu$m [MJy/sr]')
    cb.ax.tick_params(labelsize=lsz)


    ax = plt.subplot(248)
    plt.scatter(x,y,marker=mkr,c=soften(sub['GALEXFUV']*1e3),
                edgecolors='none',zorder=100,cmap=cm,s=ms)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_aspect('equal')
    cb = plt.colorbar(orientation='horizontal',pad=0.05)
    cb.set_label(r'FUV [kJy/sr]')
    cb.ax.tick_params(labelsize=lsz)

    plt.tight_layout()

    plt.savefig('ngc3627map.pdf')
    plt.close()




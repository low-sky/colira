from astropy.table import Table
import matplotlib.pyplot as p
import numpy as np
from matplotlib import rc
#rc('text', usetex=True)
rc('font',family='serif')

def pressure():
    t = Table.read('brs_category.PRESSURE.txt',format='ascii')
    
    t['R21+'] = t['R21+']-t['R21']
    t['R21-'] = t['R21']-t['R21-']
    t['R32+'] = t['R32+']-t['R32']
    t['R32-'] = t['R32']-t['R32-']

    fig = p.figure(1,figsize=(4.0,4.0))
    ax = p.subplot(111)
    ax.set_xscale('log')
    #ax.set_xlim(6e-4,1e-1)
    ax.set_ylim(0.0,1.0)
    ax.set_xlabel(r'$P/k$ (K cm$^{-3}$)')
    ax.set_ylabel(r'Line Ratio')
    p.errorbar(1e1**t['MedKey'],t['R32'],yerr=[t['R32-'],t['R32+']],\
                   marker='^',color='black',ecolor='gray',label='$R_{32}$')

    p.errorbar(1e1**t['MedKey'],t['R21'],yerr=[t['R21-'],t['R21+']],\
                   marker='o',color='black',ecolor='gray',label='$R_{21}$')
    p.tight_layout()
    #p.subplots_adjust(bottom=0.14)
    p.legend(loc=4)
    p.savefig('ratio_vs_press.pdf',bbox='tight')
    p.close()

def sfr():
    t = Table.read('brs_category.SFR.txt',format='ascii')
    
    t['R21+'] = t['R21+']-t['R21']
    t['R21-'] = t['R21']-t['R21-']
    t['R32+'] = t['R32+']-t['R32']
    t['R32-'] = t['R32']-t['R32-']

    fig = p.figure(1,figsize=(4.0,4.0))
    ax = p.subplot(111)
    ax.set_xscale('log')
    #ax.set_xlim(6e-4,1e-1)
    ax.set_ylim(0.0,1.0)
    ax.set_xlabel(r'$\Sigma_{\mathrm{SFR}}$ ($M_{\odot}$ yr$^{-1}$ kpc$^{-2}$)')
    ax.set_ylabel(r'Line Ratio')
    p.errorbar(1e1**t['MedKey'],t['R32'],yerr=[t['R32-'],t['R32+']],\
                   marker='^',color='black',ecolor='gray',label='$R_{32}$')

    p.errorbar(1e1**t['MedKey'],t['R21'],yerr=[t['R21-'],t['R21+']],\
                   marker='o',color='black',ecolor='gray',label='$R_{21}$')
    p.tight_layout()
    #p.subplots_adjust(bottom=0.14)
    p.legend(loc=4)
    p.savefig('ratio_vs_sfr.pdf',bbox='tight')
    p.close()

def uvcolor():
    t = Table.read('brs_category.UVCOLOR.txt',format='ascii')
    
    t['R21+'] = t['R21+']-t['R21']
    t['R21-'] = t['R21']-t['R21-']
    t['R32+'] = t['R32+']-t['R32']
    t['R32-'] = t['R32']-t['R32-']

    fig = p.figure(1,figsize=(4.0,4.0))
    ax = p.subplot(111)
    ax.set_xscale('log')
    #ax.set_xlim(6e-4,1e-1)
    ax.set_ylim(0.0,1.0)
    ax.set_xlabel(r'I(FUV)/I(NUV)')
    ax.set_ylabel(r'Line Ratio')
    p.errorbar(1e1**t['MedKey'],t['R32'],yerr=[t['R32-'],t['R32+']],\
                   marker='^',color='black',ecolor='gray',label='$R_{32}$')

    p.errorbar(1e1**t['MedKey'],t['R21'],yerr=[t['R21-'],t['R21+']],\
                   marker='o',color='black',ecolor='gray',label='$R_{21}$')
    p.tight_layout()
    #p.subplots_adjust(bottom=0.14)
    p.legend(loc=4)
    p.savefig('ratio_vs_uvcolor.pdf',bbox='tight')
    p.close()

def stellarsd():
    t = Table.read('brs_category.STELLARSD.txt',format='ascii')
    
    t['R21+'] = t['R21+']-t['R21']
    t['R21-'] = t['R21']-t['R21-']
    t['R32+'] = t['R32+']-t['R32']
    t['R32-'] = t['R32']-t['R32-']

    fig = p.figure(1,figsize=(4.0,4.0))
    ax = p.subplot(111)
    ax.set_xscale('log')
    #ax.set_xlim(6e-4,1e-1)
    ax.set_ylim(0.0,1.0)
    ax.set_xlabel(r'$\Sigma_\star$ ($M_{\odot}$~pc$^{-2}$)')
    ax.set_ylabel(r'Line Ratio')
    p.errorbar(1e1**t['MedKey'],t['R32'],yerr=[t['R32-'],t['R32+']],\
                   marker='^',color='black',ecolor='gray',label='$R_{32}$')

    p.errorbar(1e1**t['MedKey'],t['R21'],yerr=[t['R21-'],t['R21+']],\
                   marker='o',color='black',ecolor='gray',label='$R_{21}$')
    p.tight_layout()
    #p.subplots_adjust(bottom=0.14)
    p.legend(loc=4)
    p.savefig('ratio_vs_stellarsd.pdf',bbox='tight')
    p.close()


def rgalnorm():
    t = Table.read('brs_category.RGALNORM.txt',format='ascii')
    
    t['R21+'] = t['R21+']-t['R21']
    t['R21-'] = t['R21']-t['R21-']
    t['R32+'] = t['R32+']-t['R32']
    t['R32-'] = t['R32']-t['R32-']

    fig = p.figure(1,figsize=(4.0,4.0))
    ax = p.subplot(111)
    #ax.set_xlim(6e-4,1e-1)
    ax.set_ylim(0.0,1.0)
    ax.set_xlabel(r'$R_{gal}/R_{25}$')
    ax.set_ylabel(r'Line Ratio')
    p.errorbar(1e1**t['MedKey'],t['R32'],yerr=[t['R32-'],t['R32+']],\
                   marker='^',color='black',ecolor='gray',label='$R_{32}$')

    p.errorbar(1e1**t['MedKey'],t['R21'],yerr=[t['R21-'],t['R21+']],\
                   marker='o',color='black',ecolor='gray',label='$R_{21}$')
    p.tight_layout()
    #p.subplots_adjust(bottom=0.14)
    p.legend(loc=4)
    p.savefig('ratio_vs_rgalnorm.pdf',bbox='tight')
    p.close()

def rgal():
    t = Table.read('brs_category.RGAL.txt',format='ascii')
    
    t['R21+'] = t['R21+']-t['R21']
    t['R21-'] = t['R21']-t['R21-']
    t['R32+'] = t['R32+']-t['R32']
    t['R32-'] = t['R32']-t['R32-']

    fig = p.figure(1,figsize=(4.0,4.0))
    ax = p.subplot(111)
    #ax.set_xlim(6e-4,1e-1)
    ax.set_ylim(0.0,1.0)
    ax.set_xlabel(r'$R_{gal}$ (kpc)$')
    ax.set_ylabel(r'Line Ratio')
    p.errorbar(1e1**t['MedKey'],t['R32'],yerr=[t['R32-'],t['R32+']],\
                   marker='^',color='black',ecolor='gray',label='$R_{32}$')

    p.errorbar(1e1**t['MedKey'],t['R21'],yerr=[t['R21-'],t['R21+']],\
                   marker='o',color='black',ecolor='gray',label='$R_{21}$')
    p.tight_layout()
    #p.subplots_adjust(bottom=0.14)
    p.legend(loc=4)
    p.savefig('ratio_vs_rgalnorm.pdf',bbox='tight')
    p.close()

def ircolor():
    t = Table.read('brs_category.IRCOLOR.txt',format='ascii')
    
    t['R21+'] = t['R21+']-t['R21']
    t['R21-'] = t['R21']-t['R21-']
    t['R32+'] = t['R32+']-t['R32']
    t['R32-'] = t['R32']-t['R32-']

    fig = p.figure(1,figsize=(4.0,4.0))
    ax = p.subplot(111)
    #ax.set_xlim(6e-4,1e-1)
    ax.set_ylim(0.0,1.0)
    ax.set_xlabel(r'$I(24~\mu\mathrm{m})/I(170~\mu\mathrm{m}$)$')
    ax.set_ylabel(r'Line Ratio')
    p.errorbar(1e1**t['MedKey'],t['R32'],yerr=[t['R32-'],t['R32+']],\
                   marker='^',color='black',ecolor='gray',label='$R_{32}$')

    p.errorbar(1e1**t['MedKey'],t['R21'],yerr=[t['R21-'],t['R21+']],\
                   marker='o',color='black',ecolor='gray',label='$R_{21}$')
    p.tight_layout()
    #p.subplots_adjust(bottom=0.14)
    p.legend(loc=4)
    p.savefig('ratio_vs_ircolor.pdf',bbox='tight')
    p.close()

def spire1():
    t = Table.read('brs_category.SPIRE1.txt',format='ascii')
    t['R21+'] = t['R21+']-t['R21']
    t['R21-'] = t['R21']-t['R21-']
    t['R32+'] = t['R32+']-t['R32']
    t['R32-'] = t['R32']-t['R32-']

    fig = p.figure(1,figsize=(4.0,4.0))
    ax = p.subplot(111)
    #ax.set_xlim(6e-4,1e-1)
    ax.set_ylim(0.0,1.0)
    ax.set_xlabel(r'I(70~\mu\mathrm{m})$')
    ax.set_ylabel(r'Line Ratio')
    p.errorbar(1e1**t['MedKey'],t['R32'],yerr=[t['R32-'],t['R32+']],\
                   marker='^',color='black',ecolor='gray',label='$R_{32}$')

    p.errorbar(1e1**t['MedKey'],t['R21'],yerr=[t['R21-'],t['R21+']],\
                   marker='o',color='black',ecolor='gray',label='$R_{21}$')
    p.tight_layout()
    #p.subplots_adjust(bottom=0.14)
    p.legend(loc=4)
    p.savefig('ratio_vs_spire1.pdf',bbox='tight')
    p.close()

def multipanel():
    rc('font',size=9)
    catlist = ['RGAL','RGALNORM','SPIRE1','IRCOLOR','FUV','UVCOLOR','SFR','PRESSURE',
               'STELLARSD']
    catlabel = [r'$R_{\mathrm{gal}}$ (kpc)',
                r'$R_{\mathrm{gal}}/R_{25}$',\
                r'$I_{250}$ (MJy sr$^{-1}$)',\
                r'$I_{24}/I_{160}$',\
                r'$I_{\mathrm{FUV}}$ (mJy)',
                r'$I_{\mathrm{FUV}}/I_{\mathrm{NUV}}$',\
                r'$\Sigma_{\mathrm{SFR}}$ ($M_{\odot}$ yr$^{-1}$ kpc$^{-2}$)',\
                r'$\Sigma_{\mathrm{H2}}/\Sigma_{\mathrm{HI}}$',
                r'$\Sigma_{\star}$ ($M_{\odot}$ kpc$^{-2}$ yr$^{-1}$)']
    catfac = np.ones(9)
    catfac[0] = 1e3
    catax = ['linear','linear','log','log','log','log','log','log','log']
    fig = p.figure(1,figsize=(7.5,7.5))
    for ii,tag in enumerate(catlist):
        t = Table.read('brs_category.'+tag+'.txt',format='ascii')
#         t['R21+'] = t['R21+']-t['R21']
#         t['R21-'] = t['R21']-t['R21-']
#         t['R32+'] = t['R32+']-t['R32']
#         t['R32-'] = t['R32']-t['R32-']
        ax = p.subplot(3,3,ii+1)
        ax.set_xscale(catax[ii])
        ax.set_ylim(0.0,1.0)
        ax.set_xlabel(catlabel[ii])
        if (ii % 3) == 0:
            ax.set_ylabel(r'Line Ratio')
        p.errorbar(1e1**t['MedKey32']/catfac[ii],
                   t['R32'],yerr=[t['R32-'],t['R32+']],\
                   marker='^',color='black',ecolor='gray',label='$R_{32}$')

        p.errorbar(1e1**t['MedKey21']/catfac[ii],
                   t['R21'],yerr=[t['R21-'],t['R21+']],\
                   marker='o',color='black',ecolor='gray',label='$R_{21}$')
#
        #p.subplots_adjust(bottom=0.14)
        if ii==5:
            p.legend(loc=4)
    p.tight_layout(h_pad=0.5)    
    p.savefig('ratio_multifactor.pdf',bbox='tight')
    p.close()




    

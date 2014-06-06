from astropy.table import Table
import matplotlib.pyplot as p
import numpy as np
from matplotlib import rc
#rc('text', usetex=True)
rc('font',family='serif')

def galscatter2d():
    t = Table.read('brs.bygal2d.txt',format='ascii')

    fig = p.figure(figsize=(4.5,4.5))
    t['R21+'] = t['R21+']-t['R21']
    t['R21-'] = t['R21']-t['R21-']
    t['R32+'] = t['R32+']-t['R32']
    t['R32-'] = t['R32']-t['R32-']
    ax = p.subplot(111)
    #ax.set_xlim(6e-4,1e-1)
    ax.set_xlabel(r'$R_{21}$')
    ax.set_ylabel(r'$R_{32}$')

    marker = np.ones(len(t['Name'])).astype('str')
    marker[:]='4'
    

    p.errorbar(t['R21'],t['R32'],
               xerr=[t['R21-'],t['R21+']],
               yerr=[t['R32-'],t['R32+']],
               linestyle='None',marker=marker,color='grey',
               label=np.asarray(t['Name']))
 
    p.legend()

#     for ii,gal in enumerate(t['Name']):
#         p.errorbar(t['R21'][ii],t['R32'][ii],
#                    xerr=[t['R21-'][ii],t['R21+'][ii]],
#                    yerr=[t['R32-'][ii],t['R32+'][ii]],
#                    linestyle='None',marker='o',color='grey',label=gal)


    p.tight_layout()
    p.savefig('ratio_bygal2d.pdf',bbox='tight')
    p.close()


#!/usr/bin/env python

import scipy.stats
import numpy as np
import astropy.io.fits as fits
import emcee
import matplotlib.pyplot as p
from matplotlib import rc
from astropy.table import Table, Column
from itertools import cycle
rc('text',usetex=True)

t = Table.read('brs_bygal.RGALNORM.txt',format='ascii')
GalNames = np.unique(t['Name'])


lines = ["-","--",":"]
psym = ['p','D','o','*','v']#,'^','s']
color = ['black','gray']

linecycle = cycle(lines)
symcycle = cycle(psym)
colorcycle = cycle(color)


f = p.figure(1,figsize=(6.5,7.0))
p.subplot(211)
for name in GalNames:
    ind = np.where((t['Name']==name)&((t['R21+']-t['R21-'])<0.4)&
                   (t['Npts']>10))
    l = next(linecycle)
    ps = next(symcycle)
    c = next(colorcycle)


    if len(ind)>0:
        r = 0.5*(t['LowKey'][ind]+t['HighKey'][ind])
        if (r>0.0).any():
            p.plot(0.5*(t['LowKey'][ind]+t['HighKey'][ind]),
                   t['R21'][ind],label=name,
                   linestyle=l,marker=ps,color=c)
            
p.xlabel(r'$r_{gal}/r_{25}$')
p.ylabel(r'$R_{21}$')
p.legend(fontsize='xx-small',ncol=3,markerscale=0.7)

lines = ["-","--",":"]
psym = ['p','D','o','*','v']#,'^','s']
color = ['black','gray']

linecycle = cycle(lines)
symcycle = cycle(psym)
colorcycle = cycle(color)

p.subplot(212)
for name in GalNames:
    ind = np.where((t['Name']==name)&((t['R32+']-t['R32-'])<0.4)&
                   (t['Npts']>10))
    l = next(linecycle)
    ps = next(symcycle)
    c = next(colorcycle)
    if len(ind):
        r = 0.5*(t['LowKey'][ind]+t['HighKey'][ind])
        if (r>0.0).any():
            p.plot(r,t['R32'][ind],label=name,\
                       linestyle=l,marker=ps,color=c)

p.xlabel(r'$r_{gal}/r_{25}$')
p.ylabel(r'$R_{32}$')

p.savefig('ratio_vs_rnorm.pdf',format='pdf',bbox_inches='tight')

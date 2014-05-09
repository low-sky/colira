
from astropy.table import Table
import matplotlib.pyplot as p
import numpy as np
from matplotlib import rc
rc('text', usetex=True)

t = Table.read('brs_category.PRESSURE.txt',format='ascii')

t['xR21+'] = t['xR21+']-t['xR21']
t['xR21-'] = t['xR21']-t['xR21-']
t['xR32+'] = t['xR32+']-t['xR32']
t['xR32-'] = t['xR32']-t['xR32-']


fig = p.figure(1,figsize=(4.0,4.0))
ax = p.subplot(111)
ax.set_xscale('log')
#ax.set_xlim(6e-4,1e-1)
ax.set_ylim(0.0,1.0)
ax.set_xlabel(r'$P/k$ (K cm$^{-3}$)')
ax.set_ylabel(r'Line Ratio')
p.errorbar(1e1**t['MedKey'],t['xR32'],yerr=[t['xR32-'],t['xR32+']],\
               marker='^',color='black',ecolor='gray',label='$R_{32}$')

p.errorbar(1e1**t['MedKey'],t['xR21'],yerr=[t['xR21-'],t['xR21+']],\
               marker='o',color='black',ecolor='gray',label='$R_{21}$')
p.tight_layout()
#p.subplots_adjust(bottom=0.14)
p.legend(loc=4)
p.savefig('ratio_vs_press.pdf',bbox='tight')
p.close()

import numpy as np
import matplotlib.pyplot as p
from astropy.table import Table

t = Table.read('brs_category.SFR.txt',format='ascii')

sfr = t['MedKey'][3:]
r32 = t['R32'][3:]
perr = t['R32+'][3:]-t['R32'][3:]
merr = np.abs(t['R32-'][3:]-t['R32'][3:])

p.errorbar(np.log10(sfr),r32,yerr=(merr,perr))



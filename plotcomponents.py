import numpy as np
from pcigale import creation_modules
from pcigale.sed import SED
import matplotlib.pyplot as plt


sed = SED()
sfh = creation_modules.get_module(
    'sfhdelayed', tau_main=1000, age_main=1000, sfr_A=1, normalise="True")
sfh.process(sed)

bc03 = creation_modules.get_module(
    'bc03', imf=0, metallicity=0.02, separation_age=10)
bc03.process(sed)

nebular = creation_modules.get_module('nebular', lines_width=3000)
nebular.process(sed)

m = np.logical_and(sed.wavelength_grid > 50, sed.wavelength_grid < 1e5)
plt.plot(sed.wavelength_grid[m], sed.luminosities.sum(axis=0)[m], label='total', color='k', lw=0.3)

for i, c in enumerate(sed.contribution_names):
    plt.plot(sed.wavelength_grid[m], sed.luminosities[i, m], label=c, alpha=0.5)
    

plt.xlim(50, 1e5)
plt.yscale('log')
plt.xscale('log')
plt.xlabel("Wavelength [nm]")
plt.ylabel("Luminosity [W]")
plt.legend(title='%d Myr ago, $\\tau$=%d Myr' % (sfh.age_main, sfh.tau_main))
plt.savefig('plotcomponents.pdf')
plt.close()

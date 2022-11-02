
import numpy as np
from pcigale import creation_modules
from pcigale.sed import SED
import matplotlib.pyplot as plt


for att_level in 0.0, 0.1, 0.3, 1.0:
    sed = SED()
    sfh = creation_modules.get_module(
        'sfhdelayed', tau_main=10, age_main=1000, sfr_A=1, normalise="True")
    sfh.process(sed)

    bc03 = creation_modules.get_module(
        'bc03', imf=0, metallicity=0.02, separation_age=10)
    bc03.process(sed)

    attn = creation_modules.get_module(
        'biattenuation', **{'E(B-V)':att_level, 'E(B-V)-AGN':0.0})
    attn.process(sed)

    dust = creation_modules.get_module('galdale2014', alpha=2.0)
    dust.process(sed)

    m = np.logical_and(sed.wavelength_grid > 50, sed.wavelength_grid < 1e5)
    plt.plot(sed.wavelength_grid[m], sed.luminosities.sum(axis=0)[m], label=att_level, lw=0.3)

plt.xlim(50, 1e5)
plt.yscale('log')
plt.xscale('log')
plt.xlabel("Wavelength [nm]")
plt.ylabel("Luminosity [W]")
plt.legend(title='E(B-V)')
plt.savefig('plotattenuation.pdf')
plt.close()

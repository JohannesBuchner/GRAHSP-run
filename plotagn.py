import numpy as np
from pcigale import creation_modules
from pcigale.sed import SED
import matplotlib.pyplot as plt
from pcigale.warehouse import SedWarehouse

gbl_warehouse = SedWarehouse(uncached = ('activategtorus', 'activatepl', 'biattenuation'))

def build_grahsp_sed(
    stellar_mass=1e10, tau_main=100, age_main=3000, alpha=2.0, 
    L_AGN=1e42,
    AFeII=10, Alines=1, linewidth=5000,
    plslope=-2, plbendloc=100, plbendwidth=0.1, uvslope=0,
    ebv=0.01, ebv_agn=0.01,
    Si=-1, fcov=0.7, COOLlam=17, COOLwidth=0.45, HOTlam=2.6, HOTwidth=0.5, HOTfcov=1.0):

    modules = [
        ('sfhdelayed', dict(tau_main=tau_main, age_main=age_main)),
        ('m2005', dict(imf = 0, metallicity=0.02, separation_age = 10)),
        ('nebular', dict(logU = -2.0, f_esc = 0.0, f_dust = 0.0, lines_width = 300.0)),
        ('activate', dict(fracAGN = -1)),
        ('activatelines', dict(AFeII=AFeII, Alines=Alines, linewidth=linewidth)),
        ('activategtorus', dict(Si=Si, fcov=fcov, COOLlam=COOLlam, COOLwidth=COOLwidth, HOTlam=HOTlam, HOTwidth=HOTwidth, HOTfcov=HOTfcov)),
        ('activatepl', dict(plslope=plslope, plbendloc=plbendloc, plbendwidth=plbendwidth, uvslope=uvslope)),
    ]
    sed = SED()
    for module_name, params in modules:
        gbl_warehouse.get_module_cached(module_name, **params).process(sed)
        del module_name, params

    # select AGN components
    agn_mask = np.array(['activate' in name for name in sed.contribution_names])
    sed.luminosities[~agn_mask] *= stellar_mass
    sed.info.update({k: v * stellar_mass for k, v in sed.info.items() if k in sed.mass_proportional_info and not ('activate' in k or 'agn' in k)})
    # convert from erg/s/A at 5100A to erg/s with 5100
    # convert from erg/s to W, the luminosity unit of cigale, with 1e7
    sed.luminosities[agn_mask, :] *= L_AGN / 1e7 / 510
    sed.info.update({k: v * (L_AGN / 1e7 / 510 if 'agn.lum' in k else 1) for k, v in sed.info.items() if 'activate' in k or 'agn' in k})

    post_process_modules = [
        ('activatebol', dict()),
        ('biattenuation', {'E(B-V)':ebv, 'E(B-V)-AGN':ebv_agn, "filters":""}),
        ('galdale2014', dict(alpha=alpha)),
        ('restframe_parameters', dict(EW="K/393.4777/391/395", D4000="False", beta_calz94="True", IRX="False")),
    ]

    for module_name, params in post_process_modules:
        gbl_warehouse.get_module_cached(module_name, **params).process(sed)

    sed.luminosity = sed.luminosities.sum(0)
    return sed


def plot_sed(sed, with_components=False, **kwargs):
    m = np.logical_and(sed.wavelength_grid > 100, sed.wavelength_grid < 1e5)
    lamLlam = sed.luminosity[m] * sed.wavelength_grid[m]
    plt.plot(sed.wavelength_grid[m], lamLlam, **kwargs)
    plt.ylim(lamLlam.max() / 10000, lamLlam.max())
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Luminosity [W]")

    if with_components:
        colors = {}
        unobsc = {}
        for i, contribution_name in enumerate(sed.contribution_names):
            color = None
            if 'lines' in contribution_name.lower() or '_BC' in contribution_name:
                continue
            if 'nebular' in contribution_name: continue
            if not contribution_name.startswith('attenuation'):
                # unobscured case:
                l, = plt.plot(
                    sed.wavelength_grid[m], sed.luminosities[i, m] * sed.wavelength_grid[m],
                    alpha=0.5, ls=':')
                colors[contribution_name] = l.get_color()
                unobsc[contribution_name] = sed.luminosities[i, m]
            else:
                # total case:
                color = colors.get(contribution_name.replace('attenuation.', ''))
                plt.plot(
                    sed.wavelength_grid[m], (unobsc[contribution_name.replace('attenuation.', '')] + sed.luminosities[i, m]) * sed.wavelength_grid[m],
                    label=contribution_name.replace('attenuation.', ''), alpha=0.5, color=color, ls='-')

def main():
    sed = build_grahsp_sed(stellar_mass=1e10, age_main=500, L_AGN=1e44)
    plot_sed(sed, with_components=True, color='k', lw=1)
    plt.legend()
    plt.savefig('plotagn.pdf')
    plt.close()

if __name__ == '__main__':
    main()

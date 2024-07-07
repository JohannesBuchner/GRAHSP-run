"""

"""

import sys
import argparse
import itertools
import numpy as np
#from numpy import log, log10

import matplotlib.pyplot as plt
import matplotlib as mpl

import pcigale
from pcigale.session.configuration import Configuration
from pcigale.utils import read_table
from pcigale.data import Database
import astropy.cosmology
#import astropy.units as units
#from astropy.table import Table
import pcigale.creation_modules.redshifting

# parse command line arguments


class HelpfulParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


parser = HelpfulParser(
    description=__doc__,
    epilog="""Johannes Buchner (C) 2013-2023 <johannes.buchner.acad@gmx.com>""",
    formatter_class=argparse.RawDescriptionHelpFormatter
)


args = parser.parse_args()

# keeping it called pcigale.ini allows running pcigale with the same file
# if the user wants to run cigale
print("GRAHSP version %s | parsing pcigale.ini..." % pcigale.__version__)
config = Configuration("pcigale.ini")

data_file = config.configuration['data_file']
column_list = config.configuration['column_list']

inputs = read_table(data_file)
for c in inputs.colnames:
    inputs.rename_column(c, c + '_in')
print("%d inputs" % len(inputs))
print("reading output ...")
outputs = astropy.io.ascii.read(data_file + '_analysis_results.txt', format='commented_header')
if len(outputs) != len(inputs):
    print("WARNING: %d input rows, but %d output rows" % (len(inputs), len(outputs)))

ids_are_strings = outputs['id'].dtype.kind in 'US' or inputs['id_in'].dtype.kind in 'US'
if ids_are_strings:
    print("stripping id strings")
    inputs['id_in'] = [i.rstrip() for i in inputs['id_in']]
    outputs['id'] = [i.rstrip() for i in outputs['id']]

print("joining ...")
# join by id, write as fits file
full = astropy.table.join(
    inputs, outputs, keys_left='id_in', keys_right='id',
    table_names=['_in', ''], uniq_col_name='{col_name}{table_name}')
del full['id_in']
print("writing %s" % (data_file + '_analysis_results.fits'), "%d outputs" % len(full))
full.write(data_file + '_analysis_results.fits', overwrite=True)

print("diagnosing residuals...")
# get list of user-selected filters
filters = [name for name in column_list if not name.endswith('_err')]

plt.figure(figsize=(12, 3))
default_color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
depths = []

with Database() as base:
    for color, filtername in zip(default_color_cycle, filters):
        f = base.get_filter(filtername.rstrip('_'))
        wl_eff = f.effective_wavelength / 1000
        
        flux_obs = full[filtername + '_in']
        flux_obs_err = full[filtername + '_err_in']
        if np.any(flux_obs_err > 0):
            depths.append((filtername, color, wl_eff, np.percentile(flux_obs_err[flux_obs_err>0], 50)))
        flux_model = full['totalflux_%s_mean' % filtername]
        with np.errstate(invalid='ignore', divide='ignore'):
            delta = -2.5 * np.log10(flux_obs) - -2.5 * np.log10(flux_model + 1e-10)
        mask_good = np.isfinite(delta)
        if not mask_good.any():
            continue
        plt.text(wl_eff, 2, filtername, rotation=90, va='top', ha='center', color=color)
        props = dict(color=color)
        bplot = plt.boxplot(
            [delta[mask_good]],
            positions=[wl_eff], widths=0.1 * wl_eff,
            meanline=True, showmeans=True, showcaps=True, showbox=True,
            notch=True, sym='x', vert=True, whis=1.5, 
            showfliers=True, labels=[filtername], #patch_artist=True,
            capprops=props, boxprops=props, whiskerprops=props,
            flierprops=dict(ms=2, mew=1, mec=color, **props),
            medianprops=dict(ls='-', **props), meanprops=dict(ls=':', **props))
            #[patch.set_facecolor(color) for patch in bplot['boxes']]

#plt.yscale('log')
plt.xscale('log')
xlo, xhi = plt.xlim()
plt.xlim(xlo, xhi)
plt.hlines(0, xlo, xhi, colors=['k'], lw=1)
plt.ylim(-2, 2)
plt.ylabel(r'Observed - Model magnitude')
plt.xlabel(r'Observed Wavelength [$\mu{}m$]')

def to_mag(x):
    with np.errstate(invalid='ignore', divide='ignore'):
        return -2.5 * np.log10(x)
def from_mag(x):
    return 10**(x/-2.5)

secax = plt.gca().secondary_yaxis('right', functions=(from_mag, to_mag))
secax.set_ylabel(r'Observed / Model flux')
secax.set_yticks([0.2, 0.5, 0.8, 1, 1.2, 2, 5])

plt.gca().get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
plt.tight_layout()
plt.savefig(data_file + '_filterresiduals.pdf')
plt.close()

plt.figure(figsize=(5, 3))
for filtername, color, wl_eff, flux_lo in depths:
    plt.plot(wl_eff, -2.5 * np.log10(flux_lo / 3631000), 'o ', mfc='none', c='k')

plt.xscale('log')
plt.gca().invert_yaxis()
plt.xlabel(r'Observed Wavelength [$\mu{}m$]')
plt.ylabel('Depth (AB mag; 50%)')
plt.gca().get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
plt.tight_layout()
plt.savefig(data_file + '_filterdepth.pdf')
plt.close()

"""
GRAHSP (Grasping reliably the AGN host stellar population)

A broad-band SED fitting tool (X-ray to mid-infrared) for investigating host galaxies of
active galactic nuclei.

Features:

- Flexible empirical AGN model to avoid modelling bias
- broad and narrow line emission from AGN, in addition to accretion disk and obscurer.
- allows obscurer diversity (hotter and colder dust) and 12µm Si in emission or absorption
- allows different attenuation for AGN and host, with the more appropriate Prevot law.
- allows systematic modelling uncertainties, prevents biased overfits with overconfident errors
- allows AGN variability across assembled photometry points (L-dependent)
- allows redshift uncertainties (from photo-z)
- explores degeneracies with nested sampling Monte Carlo.

A configuration file called pcigale.ini is needed.

A data input file (typically called input.fits) is needed.
Columns:
- id
- redshift
- redshift_err (optional)
- LAGN: AGN 5100 Angstrom luminosity
- LAGN_errlo and LAGN_errhi or LAGN_err for luminosity uncertainties

The L(2-10keV) luminosity is approximately the 5100 Angstrom luminosity,
with a systematic scatter of +-0.43 dex (Koss+2017).

"""

import os
import sys
import argparse
import numpy as np
from numpy import log, log10
import warnings
from math import erf
from importlib import import_module
import multiprocessing

import scipy.stats
from scipy.constants import c
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pcigale.session.configuration import Configuration
from pcigale.analysis_modules import get_module as get_analysis_module
from pcigale.utils import read_table
from pcigale.analysis_modules import complete_obs_table
from pcigale.warehouse import SedWarehouse
from pcigale import creation_modules
from pcigale.analysis_modules.pdf_analysis import TOLERANCE
from pcigale.data import Database
from pcigale.creation_modules.biattenuation import BiAttenuationLaw
import astropy.cosmology
import astropy.units as units
from astropy.table import Table
import pcigale.creation_modules.redshifting

from ultranest import ReactiveNestedSampler
from ultranest.plot import PredictionBand
import ultranest.stepsampler
import tqdm
import getdist
import getdist.plots
import getdist.chains
getdist.chains.print_load_details = False

# some helper classes:


class DeltaDist(object):
    """Dirac Delta distribution."""

    # provides compatibility with scipy.stats distributions when a parameter is fixed
    def __init__(self, value):
        self.value = value

    def ppf(self, u):
        return self.value

    def mean(self):
        return self.value

    def std(self):
        return 0


class FastAttenuation(object):
    """Context within which the attenuation law reduces computation.

    Per-filter extinctions, and per-component extinction are skipped.
    """
    def __enter__(self):
        BiAttenuationLaw.store_filter_attenuation = False
        BiAttenuationLaw.store_component_attenuation = False

    def __exit__(self, type, value, traceback):
        BiAttenuationLaw.store_filter_attenuation = True
        BiAttenuationLaw.store_component_attenuation = True

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


parser.add_argument(
    '--offset', type=int, default=0,
    help='row in the input file to begin processing')

parser.add_argument(
    '--every', type=int, default=1,
    help='stride in the input file to process (every nth row)')

parser.add_argument(
    '--cores', type=int, default=1,
    help='number of processes to parallelise for (see joblib.Parallel)')
parser.add_argument(
    '--num-posterior-samples', type=int, default=50,
    help='number of posterior samples to analyse in post-processing')

parser.add_argument(
    '--plot', action='store_true',
    help='also make plots of the SED and parameter constraints')
parser.add_argument(
    '--mass-max', type=float, default=15,
    help='Maximum stellar mass.')
parser.add_argument(
    '--sfr-max', type=float, default=100000,
    help='Maximum SFR, averaged over the last 100Myrs, in Msun/yr.')
parser.add_argument(
    '--randomize', action='store_true',
    help='Randomize order in which to analyse observations.')

parser.add_argument(
    'action', type=str, default='analyse', choices=('analyse', 'generate-from-prior', 'plot-model', 'list-filters'),
    help='''Mode.
generate-from-prior: Generate a file with fluxes from randomly drawn model instances.
plot-model: plot model SED variations.
list-filters: list the available photometric filters, or
analyse a photometry file.''')

parser.add_argument(
    '--sampler', type=str, default='nested-slice', choices=('nested-slice',),
    help='Parameter space sampling algorithm to use. Nested-slice is recommended.')


args = parser.parse_args()

# keeping it called pcigale.ini allows running pcigale with the same file
# if the user wants to run cigale
print("parsing pcigale.ini...")
config = Configuration("pcigale.ini")

cosmo_string = config.config.get('cosmology', 'concordance')
if cosmo_string != 'concordance':
    if not hasattr(astropy.cosmology, cosmo_string):
        print("ERROR: cosmology must be set to one of: concordance, " + ', '.join(astropy.cosmology.realizations.available))
    pcigale.creation_modules.redshifting.cosmology = getattr(astropy.cosmology, cosmo_string)
cosmology = pcigale.creation_modules.redshifting.cosmology

data_file = config.configuration['data_file']
column_list = config.configuration['column_list']
module_list = config.configuration['creation_modules']
statistics_config = config.config['statistics']
# receive number of cores from command line
# configuration describes the model, command line describes how to run
n_cores = args.cores
parameter_list = config.configuration['creation_modules_params']
# limit caching to the first few modules, rest is on-the-fly
cache_depth = module_list.index('biattenuation')
cache_depth_to_clear = cache_depth
if module_list[cache_depth_to_clear - 1] == 'activatebol':
    cache_depth_to_clear -= 1
if module_list[cache_depth_to_clear - 1] == 'activatepl':
    cache_depth_to_clear -= 1
if module_list[cache_depth_to_clear - 1] == 'activategtorus':
    cache_depth_to_clear -= 1
cache_max = int(os.environ.get('CACHE_MAX', '10000'))
#chunk_size = int(os.environ.get('CHUNKSIZE', '20'))
mp_ctx = multiprocessing.get_context(os.environ.get('MP_METHOD', 'forkserver'))
cache_print = os.environ.get('CACHE_VERBOSE', '0') == '1'
if cache_print:
    print("Caching modules:", module_list[:cache_depth])
    print("Caching SEDs:", module_list[:cache_depth_to_clear], "with %d entries" % cache_max)
analysis_module = get_analysis_module(config.configuration[
    'analysis_method'])
analysis_module_params = config.configuration['analysis_method_params']

analysed_variables = analysis_module_params["analysed_variables"]
n_variables = len(analysed_variables)
lim_flag = analysis_module_params["lim_flag"].lower() == "true"
mock_flag = analysis_module_params["mock_flag"].lower() == "true"

# get statistics configuration
exponent = int(statistics_config.get('exponent', '2'))
with_attenuation_model_uncertainty = statistics_config.get('attenuation_model_uncertainty', 'false').lower() == 'true'
variability_uncertainty = statistics_config.get('variability_uncertainty', 'true').lower() == 'true'
systematics_width = float(statistics_config.get('systematics_width', '0.01'))

gbl_warehouse = SedWarehouse(store_depth=cache_depth, reusable=('activategtorus', 'activatepl', 'activatebol', 'biattenuation'))

def list_filters():
    """List all known filters."""
    with Database() as base:
        keys, _ = base.get_filter_list()
        for filter_name in keys:
            f = base.get_filter(filter_name)
            print(f)


# get list of user-selected filters
filters = [name for name in column_list if not name.endswith('_err')]
with Database() as base:
    filters_wl_orig = np.array([base.get_filter(name.rstrip('_')).effective_wavelength for name in filters])
n_filters = len(filters)

# show chosen parameters to the user, in latex file and screen
latex_table = open('pcigale.ini.tex', 'w')
latex_table.write(r'  Parameter & Description & Values \\' + "\n")
latex_table.write(r'  \hline' + "\n")
latex_table.write(r'  \hline' + "\n")
latex_table.write(r'  Galaxy components: & & \\' + "\n")
latex_table.write(r'  \texttt{stellar\_mass} & & log-uniform between $10^5$ and $10^{\mathtt{mass\_max}} M_\odot$ \\' + "\n")
latex_table.write(r'  \texttt{mass\_max} & & %d \\' % (args.mass_max) + "\n")
param_names = []
is_log_param = []
print()
print("Parameters")
print("----------")
for module_name, module_parameters in zip(module_list, parameter_list):
    print("  [%s]" % module_name)
    latex_table.write("  \\texttt{[%s]} & & \\\\\n" % module_name)
    module = import_module("." + module_name, 'pcigale.creation_modules')
    for k, v in module_parameters.items():
        description = module.Module.parameter_list[k][1].split('.')[0]
        if len(v) > 1:
            if min(v) > 0 and max(v) > 0 and max(v) / min(v) > 40:
                is_log_param.append(True)
                param_names.append("log_%s_%s" % (module_name, k))
            else:
                is_log_param.append(False)
                param_names.append("%s.%s" % (module_name, k))
            print("    %20s : %s" % (k, v), '(log-uniform)' if is_log_param[-1] else '(uniform)')
            latex_table.write("  \\texttt{%s} & %s & %s %s \\\\\n" % (
                k.replace('_', '\\_'), description.replace('&', '\\&').replace('_', '\\_'),
                str(v).replace('[', '').replace(']', ''),
                '(log-uniform)' if is_log_param[-1] else '(uniform)')
            )
        else:
            print("    %20s = %s" % (k, v[0]))
            latex_table.write("  \\texttt{%s} & %s & %s %s \\\\\n" % (
                k.replace('_', '\\_'), description.replace('&', '\\&').replace('_', '\\_'),
                v[0], '(fixed)'
            ))
        del k, v

    del module_name, module_parameters
    latex_table.write(r'  \hline' + "\n")
print()
latex_table.write(r'  \hline' + "\n")
latex_table.close()
param_names.append("log_stellar_mass")
param_names.append("log_L_AGN")
param_names.append("redshift")
param_names.append("systematics")
rv_systematics = scipy.stats.expon(scale=systematics_width)

print("Statistics")
print("----------")
print(" Exponent: %d (%s)" % (exponent, {1: 'L1', 2: 'Gaussian'}[exponent]))
print(" model uncertainty: ")
print("    white: exponential, scale=%s" % systematics_width)
print("    attenuation: %s" % with_attenuation_model_uncertainty)
print("    variability: %s" % variability_uncertainty)
print()
print("Cosmology:", cosmology)
print()

def make_parameter_list(parameters):
    """Make a parameter list given a array of values, which may be in log."""
    parameter_list_first = []
    i = 0
    for module_parameters in parameter_list:
        parameter_list_here = {}
        for k, v in module_parameters.items():
            if len(v) == 1:
                parameter_list_here[k] = v[0]
            else:
                if is_log_param[i]:
                    parameter_list_here[k] = 10**(parameters[i])
                else:
                    parameter_list_here[k] = parameters[i]
                i += 1

        parameter_list_first.append(parameter_list_here)
    return parameter_list_first


def compute_model_fluxes(sed, filters):
    """Compute fluxes and derived properties.

    Returns -99 values if the setup is unphysical (star formation before the age of the Universe).
    """
    if 'sfh.age' in sed.info and sed.info['sfh.age'] > sed.info['universe.age']:
        model_fluxes = -99. * np.ones(len(filters))
        model_variables = -99. * np.ones(len(analysed_variables))
    else:
        model_fluxes = np.array([sed.compute_fnu(filter_.rstrip('_')) for filter_ in filters])
        model_variables = np.array([sed.info[name] for name in analysed_variables])

    return model_fluxes, model_variables


def scale_sed_components(module_list, parameter_list_here, stellar_mass, L_AGN):
    """Create the hybrid galaxy & AGN SED.

    Runs the dual pipelines,
     - one for AGN components with mock galaxy properties
     - one for galaxy components with mock galaxy properties
    Then combines the result, with the SED scaled by stellar mass or AGN luminosity.

    Returns the scaled SED, and the unscaled galaxy and AGN SED
    """
    # get sed for galaxy
    parameter_list_gal = []
    parameter_list_agn = []
    for i, (module_name, module_parameters_available, module_parameters_selected) in enumerate(zip(module_list, parameter_list, parameter_list_here)):
        parameter_list_gal_here = {}
        parameter_list_agn_here = {}
        for k in sorted(module_parameters_available.keys()):
            selected_value = module_parameters_selected[k]
            mock_value = module_parameters_available[k][0]
            if i >= cache_depth:
                # use the true value in both cases, because:
                # module applies to both and is not cached
                parameter_list_gal_here[k] = selected_value
                parameter_list_agn_here[k] = selected_value
            elif 'activate' in module_name or 'AGN' in k:
                # mock the value for galaxy sed
                parameter_list_gal_here[k] = mock_value
                parameter_list_agn_here[k] = selected_value
            else:
                # mock the value for AGN sed
                parameter_list_gal_here[k] = selected_value
                parameter_list_agn_here[k] = mock_value
            del k
        parameter_list_gal.append(parameter_list_gal_here)
        parameter_list_agn.append(parameter_list_agn_here)

    # this clears the cache, if we are in danger of running out of memory (CACHE_MAX environment variable)
    if len(gbl_warehouse.storage.dictionary) > cache_max:
        if cache_print:
            sys.stderr.write("clearing cache (%d objects) ..." % len(gbl_warehouse.storage.dictionary))
        if np.random.uniform() < 0.01:
            # clear cache completely, occasionally, for speed
            gbl_warehouse.partial_clear_cache(0)
        else:
            gbl_warehouse.partial_clear_cache(cache_depth_to_clear)
        if cache_print:
            sys.stderr.write("cleared, %d remain.  \n" % len(gbl_warehouse.storage.dictionary))

    # compute galaxy and AGN SEDs, un-normalised
    sed = gbl_warehouse.get_sed(module_list[:cache_depth], parameter_list_gal[:cache_depth])
    agn_sed = gbl_warehouse.get_sed(module_list[:cache_depth], parameter_list_agn[:cache_depth]).copy()
    assert sed.contribution_names == agn_sed.contribution_names, (sed.contribution_names, agn_sed.contribution_names)

    # select AGN components
    agn_mask = np.array(['activate' in name for name in sed.contribution_names])

    # scale the AGN and galactic components as needed
    scaled_sed = sed.copy()
    scaled_sed.luminosities[~agn_mask] *= stellar_mass
    scaled_sed.info.update({k: v * stellar_mass for k, v in sed.info.items() if k in sed.mass_proportional_info})
    # convert from erg/s/A at 5100A to erg/s with 5100
    # convert from erg/s to W, the luminosity unit of cigale, with 1e7
    agn_sed.luminosities[agn_mask, :] *= L_AGN / 1e7 / 510
    scaled_sed.luminosities[agn_mask] = agn_sed.luminosities[agn_mask]
    agn_sed.luminosity = agn_sed.luminosities[agn_mask, :].sum(axis=0)
    # copy over AGN meta data
    scaled_sed.info.update({k: v * (L_AGN if 'agn.lum' in k else 1) for k, v in agn_sed.info.items() if 'activate' in k or 'agn' in k})
    agn_sed.info.update({k: v * (L_AGN if 'agn.lum' in k else 1) for k, v in agn_sed.info.items() if 'activate' in k or 'agn' in k})
    scaled_sed.luminosity = scaled_sed.luminosities.sum(0)

    # apply the remaining modules (post-caching)
    for module_name, module_parameters in zip(module_list[cache_depth:], parameter_list_here[cache_depth:]):
        module_instance = gbl_warehouse.get_module_cached(module_name, **module_parameters)
        module_instance.process(scaled_sed)

    return scaled_sed, sed, agn_sed


# Wavelength limits (restframe) when plotting the best SED.
PLOT_L_MIN = 0.1
PLOT_L_MAX = 50


def plot_posteriors(filename, prior_samples, param_names, samples):
    """Make plot of parameter posteriors compared to prior."""
    print("plotting posteriors ...")
    plt.figure(figsize=(12, 12))
    for i, (param_name, samples) in enumerate(zip(param_names, samples.transpose())):
        plt.subplot(4, len(param_names) // 4 + 1, i + 1)
        bins = np.unique(list(set(prior_samples[:, i]).union(set(samples))))
        if not np.isfinite(bins).all():
            print("WARNING: parameter %s is bad, remove it from the analysis list" % param_name)
        if len(bins) > 2 and bins[-1] > bins[-2]:
            bins = np.concatenate((bins, [bins[-1] + bins[-1] - bins[-2]]))
        if len(bins) > 40:
            bins = 20
        with np.errstate(invalid='ignore', divide='ignore'):
            plt.hist(samples, histtype='step', density=True, bins=bins)
        xlo, xhi = plt.xlim()
        with np.errstate(invalid='ignore', divide='ignore'):
            plt.hist(
                prior_samples[:, i], histtype='step',
                density=True, bins=bins, color='gray', ls='-')
        plt.xlim(xlo, xhi)
        plt.yticks([])
        plt.xlabel(param_names[i])

    plt.subplots_adjust(wspace=0.1)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def _with_attenuation(keys):
    return keys + ['attenuation.' + key for key in keys]


# groups of SED contributions to include in the fit
plot_elements = [
    dict(keys=_with_attenuation(['stellar.young', 'stellar.old']),
         label="Stellar (attenuated)", color='orange', marker=None, linestyle='-',),
    # dict(keys=['stellar.young', 'stellar.old'],
    #      label="Stellar (unattenuated)", color='b', marker=None, nonposy='clip', linestyle='--', linewidth=0.5),
    dict(keys=_with_attenuation(['nebular.lines_young', 'nebular.lines_old', 'nebular.continuum_young', 'nebular.continuum_old']),
         label="Nebular emission", color='y', marker=None, linewidth=.5),
    dict(keys=_with_attenuation(['agn.activate_Disk']),
         label="AGN disk", color=[0.90, 0.90, 0.72], marker=None, linestyle='-', linewidth=1.5),
    dict(keys=_with_attenuation(['agn.activate_Torus', 'agn.activate_TorusSi']),
         label="AGN torus", color=[0.90, 0.77, 0.42], marker=None, linestyle='-', linewidth=1.5),
    dict(keys=_with_attenuation(['agn.activate_EmLines_BL', 'agn.activate_EmLines_NL', 'agn.activate_FeLines', 'agn.activate_EmLines_LINER']),
         label="AGN lines", color=[0.90, 0.50, 0.21], marker=None, linestyle='-', linewidth=0.5),
    dict(keys=['dust'], label="Dust", color='darkred', marker=None, linestyle='-', linewidth=0.5),
]


def plot_results(sampler, prior_samples, obs, obs_fluxes, obs_errors, wobs, cache_filters, replot):
    """Make all the plots."""

    # only allow the main process to plot
    if not sampler.log:
        return

    results = sampler.results
    Z = results['logz']
    plot_dir = sampler.logs['plots']
    assert np.isfinite(results['samples']).all()

    # avoid replotting if nothing changed and all the files are there
    if not replot and all((os.path.exists('%s/%s.pdf' % (plot_dir, f)) for f in ('posteriors', 'derived', 'sed_mJy', 'sed_lum'))):
        print("not replotting.")
        return

    plot_posteriors('%s/posteriors.pdf' % plot_dir, prior_samples, param_names, results['samples'])

    """
    print("making corner plot ...")
    smooth_samples = sampler.results['samples'].copy()
    for i in range(len(param_names)):
        bins = np.unique(prior_samples[:, i]).tolist()
        if len(bins) < 40:
            if len(bins) == 1:
                db = 1
            else:
                db = (bins[-1] - bins[-2])
            for lo, hi in zip(bins, bins[1:] + [bins[-1] + db]):
                mask = smooth_samples[:, i] == lo
                smooth_samples[mask, i] = np.random.uniform(lo, hi, size=mask.sum())
                #mask2 = prior_samples[:, i] == lo

    samples = getdist.MCSamples(
        samples=smooth_samples, names=param_names, sampler='nested',
        settings=dict(smooth_scale_2D=0.3, smooth_scale_1D=0.3))
    g = getdist.plots.get_subplot_plotter()
    g.triangle_plot([samples])
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    plt.savefig('%s/corner.pdf' % plot_dir)
    plt.close()
    """

    print("making SED instances for plotting ...")
    bands = {'lum': {}, 'mJy': {}}

    z = obs['redshift']
    chi2_best = 1e300

    posteriors_names = analysed_variables + ['chi2']
    stellar_mass_column = []
    posteriors = []
    all_mod_fluxes = []
    agn_mod_fluxes = []
    gal_mod_fluxes = []
    obs_filter_wavelength = filters_wl_orig[wobs]

    for parameters in tqdm.tqdm(sampler.results['samples'][:args.num_posterior_samples, :]):
        stellar_mass = 10**parameters[-4]
        L_AGN = 10**parameters[-3]
        redshift = parameters[-2]
        sys_error = parameters[-1]
        parameter_list_here = make_parameter_list(parameters)
        parameter_list_here[-1] = dict(redshift=redshift)

        sed, gal_sed, agn_sed = scale_sed_components(module_list, parameter_list_here, stellar_mass, L_AGN)
        sed.cache_filters = cache_filters
        agn_sed.cache_filters = cache_filters
        gal_sed.cache_filters = cache_filters

        model_fluxes_full, model_variables = compute_model_fluxes(sed, filters)
        for module_name, module_parameters in zip(module_list[cache_depth:], parameter_list_here[cache_depth:]):
            module_instance = gbl_warehouse.get_module_cached(module_name, **module_parameters)
            module_instance.process(agn_sed)

        agn_model_fluxes_full, _ = compute_model_fluxes(agn_sed, filters)
        agn_model_fluxes = agn_model_fluxes_full[wobs]

        mod_fluxes = model_fluxes_full[wobs]
        all_mod_fluxes.append(model_fluxes_full)
        agn_mod_fluxes.append(agn_model_fluxes_full)
        gal_mod_fluxes.append(model_fluxes_full - agn_model_fluxes_full)

        filter_wl_indices = np.searchsorted(sed.wavelength_grid, filters_wl_orig[wobs])
        filter_contrib = sed.luminosities[:, filter_wl_indices]
        filter_pos_contrib = np.where(filter_contrib > 0, filter_contrib, 0).sum(axis=0)
        transmitted_fraction = filter_contrib.sum(axis=0) / filter_pos_contrib

        _, chi2_0, _ = chi2_with_norm(
            mod_fluxes, agn_model_fluxes*0, obs_fluxes, obs_errors, obs_filter_wavelength * np.inf, redshift, sys_error * 0,
            NEV=sed.info['agn.NEV'], exponent=exponent, transmitted_fraction=transmitted_fraction * 0 + 1)
        chi2_best = min(chi2_0, chi2_best)
        norm, chi2_, total_variance = chi2_with_norm(
            mod_fluxes, agn_model_fluxes, obs_fluxes, obs_errors, obs_filter_wavelength, redshift, sys_error,
            NEV=sed.info['agn.NEV'], exponent=exponent, transmitted_fraction=transmitted_fraction)
        posteriors.append(np.concatenate((np.log10(model_variables), [chi2_])))
        stellar_mass_column.append(stellar_mass)

        wavelength_spec = sed.wavelength_grid
        DL = sed.info['universe.luminosity_distance']

        for sed_type in 'mJy', 'lum':
            wavelength_spec2 = wavelength_spec.copy()
            wavelength_spec_mask = np.logical_and(
                wavelength_spec2 >= PLOT_L_MIN * 0.9,
                wavelength_spec2 <= PLOT_L_MAX * 1.1)
            if sed_type == 'lum':
                sed_multiplier = wavelength_spec2.copy() * (redshift + 1)
                wavelength_spec2 /= 1. + z
            elif sed_type == 'mJy':
                sed_multiplier = (wavelength_spec2 * 1e29 /
                                   (c / (wavelength_spec2 * 1e-9)) /
                                   (4. * np.pi * DL * DL))

            assert (sed_multiplier >= 0).all(), (stellar_mass, DL, wavelength_spec2)
            wavelength_spec2 /= 1000

            for j, plot_element in enumerate(plot_elements):
                keys = plot_element['keys']
                if not any(k in sed.contribution_names for k in keys):
                    continue
                if j not in bands[sed_type]:
                    # print("  building", sed_type, plot_element['label'])
                    bands[sed_type][j] = PredictionBand(wavelength_spec2[wavelength_spec_mask])
                pred = sum(sed.get_lumin_contribution(k) * sed_multiplier for k in keys if k in sed.contribution_names)
                assert np.isfinite(pred).all(), pred
                assert bands[sed_type][j].x.shape == pred.shape, (bands[sed_type][j].x.shape, pred.shape)
                # print(plot_element['label'], pred)
                bands[sed_type][j].add(pred[wavelength_spec_mask])
            if 'total' not in bands[sed_type]:
                bands[sed_type]['total'] = PredictionBand(wavelength_spec2[wavelength_spec_mask])
            bands[sed_type]['total'].add((sed.luminosity * sed_multiplier)[wavelength_spec_mask])

    posteriors = np.array(posteriors)
    # add specific (normalised by stellar mass) AGN luminosities
    for i, n in enumerate(list(posteriors_names)):
        if 'sfh.sfr' in n:
            posteriors = np.hstack((posteriors, (posteriors[:,i] / stellar_mass_column).reshape((-1,1))))
            posteriors_names.append('s_' + n)
        elif 'agn.lum' in n or 'Lbol' in n:
            posteriors = np.hstack((posteriors, (posteriors[:,i] - np.log10(stellar_mass_column)).reshape((-1,1))))
            posteriors_names.append('s_' + n)
    # print("   +specific:", len(posteriors_names), posteriors_names)
    plot_posteriors('%s/derived.pdf' % plot_dir, np.zeros((0, len(posteriors_names))), posteriors_names, posteriors)
    # add model fluxes as output columns
    posteriors = np.hstack((posteriors, all_mod_fluxes, agn_mod_fluxes, gal_mod_fluxes))
    posteriors_names += ['totalflux_' + filtername for filtername in filters]
    posteriors_names += ['AGNflux_' + filtername for filtername in filters]
    posteriors_names += ['GALflux_' + filtername for filtername in filters]
    # print("   +fluxes:", len(posteriors_names), posteriors_names)
    np.savetxt('%s/derived.csv' % plot_dir, posteriors, header=','.join(posteriors_names), comments='', delimiter=',')

    print("write out SED as csv files ...")
    for sed_type in 'mJy', 'lum':
        header = ['wavelength']
        seddata = [bands[sed_type]['total'].x]
        for j, plot_element in enumerate(plot_elements):
            if j not in bands[sed_type]:
                continue
            k = plot_element['label']
            header += [k, k + '_errup', k + '_errlo']
            seddata.append(bands[sed_type][j].get_line())
            seddata.append(bands[sed_type][j].get_line(0.5 + 0.341))
            seddata.append(bands[sed_type][j].get_line(0.5 - 0.341))
        k = 'total'
        header += [k, k + '_errup', k + '_errlo']
        seddata.append(bands[sed_type][k].get_line())
        seddata.append(bands[sed_type][k].get_line(0.5 + 0.341))
        seddata.append(bands[sed_type][k].get_line(0.5 - 0.341))
        np.savetxt('%s/sed_%s.csv.gz' % (plot_dir, sed_type), np.transpose(seddata), header=','.join(header), comments='', delimiter=',')
        del sed_type

    for sed_type in 'mJy', 'lum':
        print("  plotting", sed_type)
        filters_wl = filters_wl_orig[wobs] / 1000
        # wsed = np.where((wavelength_spec2 > xmin) & (wavelength_spec2 < xmax))

        figure = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.02)

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        plt.sca(ax1)
        bands[sed_type]['total'].shade(0.45, color='k', alpha=0.2)
        bands[sed_type]['total'].line(color='k', label='Model spectrum', linewidth=1.5)
        for j, plot_element in enumerate(plot_elements):
            if j in bands[sed_type]:
                # print("  plotting", sed_type, plot_element['label'], np.shape(bands[sed_type][j].ys))
                bands[sed_type][j].shade(0.45, color=plot_element['color'], alpha=0.1)
                line_kwargs = dict(plot_element)
                del line_kwargs['keys']
                bands[sed_type][j].line(**line_kwargs)

        if sed_type == 'lum':
            # shift back from observed-frame to rest-frame units:
            xmin = PLOT_L_MIN / (1 + z)
            xmax = PLOT_L_MAX / (1 + z)

            filters_wl /= 1. + z
            k_corr_SED = 1e-29 * (4. * np.pi * DL * DL) * c / (filters_wl * 1e-9) / 1000
            obs_fluxes = obs_fluxes * k_corr_SED
            obs_fluxes_err = obs_errors * k_corr_SED
            mod_fluxes = mod_fluxes * k_corr_SED
            mod_fluxes_err = total_variance**0.5 * k_corr_SED
        elif sed_type == 'mJy':
            xmin = PLOT_L_MIN
            xmax = PLOT_L_MAX

            k_corr_SED = 1.
            obs_fluxes_err = obs_errors
            mod_fluxes_err = total_variance**0.5

        ax1.set_autoscale_on(False)
        ax1.scatter(filters_wl, mod_fluxes, marker='o', color='r', s=8,
                    zorder=3, label="Model fluxes")
        mask_ok = np.logical_and(obs_fluxes > 0., obs_errors > 0.)
        ax1.errorbar(filters_wl[mask_ok], obs_fluxes[mask_ok],
                     yerr=mod_fluxes_err[mask_ok] * 3, ls='',
                     markersize=6, color='b', capsize=2., elinewidth=1)
        ax1.errorbar(filters_wl[mask_ok], obs_fluxes[mask_ok],
                     yerr=obs_fluxes_err[mask_ok] * 3, ls='', marker='s',
                     label='Observed fluxes', markerfacecolor='None',
                     markersize=6, color='b', capsize=4., elinewidth=1)
        mask_uplim = np.logical_and(np.logical_and(obs_fluxes > 0.,
                                               obs_fluxes_err < 0.),
                                obs_fluxes_err > -9990. * k_corr_SED)

        if not mask_uplim.any() == False:
            ax1.errorbar(filters_wl[mask_uplim], obs_fluxes[mask_uplim],
                         yerr=obs_fluxes_err[mask_uplim], ls='',
                         marker='v', label='Observed upper limits',
                         markerfacecolor='None', markersize=6,
                         markeredgecolor='g',
                         capsize=2, linestyle=' ', elinewidth=1)
        mask_noerr = np.logical_and(obs_fluxes > 0.,
                                    obs_fluxes_err < -9990. * k_corr_SED)
        if not mask_noerr.any() == False:
            ax1.errorbar(filters_wl[mask_noerr], obs_fluxes[mask_noerr],
                         ls='', marker='s', markerfacecolor='None',
                         markersize=6, markeredgecolor='r',
                         label='Observed fluxes, no errors',
                         capsize=2, linestyle=' ', elinewidth=1)
        mask, = np.where(obs_fluxes > 0.)
        ax2.errorbar(filters_wl[mask],
                     (obs_fluxes[mask]-mod_fluxes[mask])/obs_fluxes[mask],
                     yerr=obs_fluxes_err[mask]/obs_fluxes[mask],
                     marker='x', color='k',
                     capsize=2, linestyle=' ', elinewidth=1)
        if mask.any():
            maxresid = max(1, max(np.abs((obs_fluxes[mask]-mod_fluxes[mask])/obs_fluxes[mask])))
        else:
            maxresid = 3.0
        ax2.plot([xmin, xmax], [0., 0.], ls='--', color='k')
        ax2.set_xscale('log')
        ax1.set_xscale('log')
        ax2.minorticks_on()

        figure.subplots_adjust(hspace=0.2, wspace=0.)

        ax1.set_xlim(xmin, xmax)
        ax2.set_xlim(xmin, xmax)
        if mask_ok.any():
            ymin = min(np.nanmin(obs_fluxes[mask_ok]),
                       np.nanmin(mod_fluxes[mask_ok]))

            if not mask_uplim.any() == False:
                ymax = max(max(np.nanmax(obs_fluxes[mask_ok]),
                               np.nanmax(obs_fluxes[mask_uplim])),
                           max(np.nanmax(mod_fluxes[mask_ok]),
                               np.nanmax(mod_fluxes[mask_uplim])))
            else:
                ymax = max(np.nanmax(obs_fluxes[mask_ok]),
                           np.nanmax(mod_fluxes[mask_ok]))
            if np.isinf(ymax):
                ymax = 100 * ymin
            if np.isinf(ymin) or ymin < 1e-6 * ymax:
                ymin = ymax / 100
            ax1.set_ylim(1e-2*ymin, 1e1*ymax)
        ax1.set_yscale('log')
        ax2.set_ylim(-maxresid, maxresid)
        if sed_type == 'lum':
            ax2.set_xlabel("Rest-frame wavelength [$\\mu$m]")
            ax1.set_ylabel("Luminosity [W]")
            ax2.set_ylabel("(Obs-Mod)/Obs", size=8)
        else:
            ax2.set_xlabel("Observed wavelength [$\\mu$m]")
            ax1.set_ylabel("Flux [mJy]")
            ax2.set_ylabel("(Obs-Mod)/Obs", size=8)
        ax1.legend(fontsize=6, loc='best', fancybox=True, framealpha=0.5)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels()[1], visible=False)
        figure.suptitle(
            "%s at z=%.3f, $\chi^2_{/n}$=%.1f/%d Z=%.1f" %
            (obs['id'], obs['redshift'], chi2_best, len(obs_fluxes), Z))
        figure.savefig("%s/sed_%s.pdf" % (plot_dir, sed_type))
        plt.close(figure)

    return (
        param_names + posteriors_names,
        np.concatenate((results['samples'].mean(axis=0), np.mean(posteriors, axis=0))),
        np.concatenate((results['samples'].std(axis=0), np.std(posteriors, axis=0))),
        np.concatenate((np.quantile(results['samples'], 0.02275, axis=0), np.quantile(posteriors, 0.02275, axis=0))),
        np.concatenate((np.quantile(results['samples'], 0.97725, axis=0), np.quantile(posteriors, 0.97725, axis=0))),
        np.concatenate((np.median(results['samples'], axis=0), np.median(posteriors, axis=0))),
        np.concatenate((np.log(np.exp(results['samples']).mean(axis=0)), np.log(np.exp(posteriors).mean(axis=0)))),
    )


def make_prior_transform(rv_redshift, Finfo=None, num_redshift_points=40):
    """Create the prior transform given prior information about the flux or redshift.

    Parameters
    ----------
    rv_redshift: scipy.stats random variable
        prior for redshift
    Finfo: None or tuple
        If None, a flat prior on AGN luminosity is assumed
        If (FAGN, FAGN_errlo, FAGN_errhi) is provided,
        this describes the log10(erg/s/cm^2 flux) prior and its error bars.
    num_redshift_points: int
        Number of discrete points for the redshift parameter.

    Returns
    -------
    prior_transform: func
        Prior transform function
    """
    redshift_fixed = rv_redshift.std() == 0

    # consider flux prior on 5100A luminosity
    if Finfo is not None:
        FAGN, FAGN_errlo, FAGN_errhi = Finfo
        # Sides of the asymmetric gaussian prior on log(flux)
        rv_F_lo = scipy.stats.norm(FAGN, FAGN_errlo)
        rv_F_hi = scipy.stats.norm(FAGN, FAGN_errhi)

        def L_prior_transform(u, redshift):
            # pick the appropriate side
            rv = rv_F_lo if u < 0.5 else rv_F_hi
            # convert to flux
            logF = rv.ppf(u)
            # convert from erg/s/cm^2 to erg/s
            logL = logF + np.log10(4 * np.pi) + 2 * np.log10((cosmology.luminosity_distance(redshift) / units.cm).to(1))
            return logL
    else:
        def L_prior_transform(u, z):
            # AGN luminosity from 10^38 to 10^50
            del z
            return u * 12 + 38

    def prior_transform(cube):
        params = np.empty(len(cube) + (1 if redshift_fixed else 0)) + np.nan
        i = 0
        for module_parameters in parameter_list:
            for k, v in module_parameters.items():
                if len(v) > 1:
                    params[i] = v[min(len(v) - 1, int(len(v) * cube[i]))]
                    if is_log_param[i]:
                        params[i] = log10(params[i])
                    i += 1

        # stellar mass from 10^5 to 10^15
        params[i] = cube[i] * (args.mass_max - 5) + 5

        # redshift.
        # Approximate redshift with points on the CDF
        params[i + 2] = rv_redshift.ppf((1 + np.round(cube[i + 2] * num_redshift_points)) / (num_redshift_points + 2))

        # AGN luminosity from 10^30 to 10^50
        params[i + 1] = L_prior_transform(cube[i + 1], params[i + 2])

        j = i + 2 if redshift_fixed else i + 3

        # systematic uncertainty
        params[i + 3] = rv_systematics.ppf(cube[j])
        return params
    return prior_transform


def plot_model():
    """Vary each model parameter and show its effect as plots.

    This also saves the plot elements as machine-readable csv files.
    """
    umid = np.array([1e-6 if 'E(B-V)' in p else 0.5 for p in param_names])
    redshift = 1.0
    for logL_AGN in [44, 38, 46, 42]:
        L_AGN = 10**logL_AGN
        rv_redshift = scipy.stats.uniform(redshift, redshift + 1e-3)
        prior_transform = make_prior_transform(rv_redshift)
        print("reference values:")
        for p, v in zip(param_names, prior_transform(umid)):
            print("   %-20s: %s" % (p, v))
        cache_filters = {}

        for i, p in enumerate(param_names):
            if p in ('redshift', 'log_L_AGN', 'activate_AGNtype', 'systematics'):
                continue
            print("varying", p)

            u = umid.copy()
            for AGNtype in 1,: # 2, 3:
                last_value = np.nan
                plt.figure(figsize=(12, 6))
                filename = 'modelspectrum_L%d_type%s_%s' % (logL_AGN, AGNtype, p.replace('(', '').replace(')', ''))
                first_legend = None
                for v in np.linspace(0.001, 0.999, 11):
                    u[i] = v
                    parameters = prior_transform(u)
                    if 'activate_AGNtype' in param_names:
                        parameters[param_names.index('activate_AGNtype')] = AGNtype
                    if parameters[i] == last_value:
                        continue
                    last_value = parameters[i]
                    stellar_mass = 10**parameters[-4]

                    parameter_list_here = make_parameter_list(parameters)
                    assert module_list[-1] == 'redshifting'
                    parameter_list_here[-1] = dict(redshift=redshift)

                    with np.errstate(invalid='ignore'):
                        sed, _, agn_sed = scale_sed_components(module_list, parameter_list_here, stellar_mass, L_AGN)
                    sed.cache_filters = cache_filters

                    _, model_variables = compute_model_fluxes(sed, filters)

                    wavelength_spec = sed.wavelength_grid.copy()
                    wavelength_spec /= 1. + redshift
                    DL = sed.info['universe.luminosity_distance']
                    sed_multiplier = wavelength_spec.copy() * (redshift + 1)
                    assert (sed_multiplier >= 0).all(), (stellar_mass, DL, wavelength_spec)
                    wavelength_spec /= 1000
                    mask = np.logical_and(wavelength_spec >= PLOT_L_MIN, wavelength_spec <= PLOT_L_MAX)
                    alpha = 1 - (v + 0.2) / 1.2

                    output_labels = ['wavelength', 'total']
                    outputs = [wavelength_spec, sed.luminosity * sed_multiplier]
                    # print(sed.contribution_names)
                    for j, plot_element in enumerate(plot_elements):
                        keys = plot_element['keys']
                        if not any(k in sed.contribution_names for k in keys):
                            # print("skipping", plot_element['label'], 'because need', keys, "have only some:", [k in sed.contribution_names for k in keys])
                            continue

                        pred = sum(sed.get_lumin_contribution(k) * sed_multiplier for k in keys if k in sed.contribution_names)

                        line_kwargs = dict(plot_element)
                        del line_kwargs['keys']
                        line_kwargs['alpha'] = alpha
                        plt.plot(wavelength_spec[mask], pred[mask], **line_kwargs)
                        outputs.append(pred)
                        output_labels.append(plot_element['label'])

                    line, = plt.plot(wavelength_spec[mask], (sed.luminosity * sed_multiplier)[mask], '-', color='k', alpha=alpha, label='total')
                    subfilename = filename + '_at%s' % (parameters[i])
                    np.savetxt(
                        subfilename + '.params',
                        [np.concatenate((parameters, model_variables))],
                        delimiter=',', header=','.join(param_names + analysed_variables)
                    )
                    np.savetxt(subfilename + '.csv', np.transpose(outputs), delimiter=',', header=','.join(output_labels), comments='')

                    if first_legend is None:
                        first_legend = plt.legend(title='Components', framealpha=0.5, loc='upper left')
                plt.gca().add_artist(first_legend)

                plt.xlabel("Wavelength [$\mu$m]")
                plt.ylabel("Luminosity [W]")
                plt.xscale('log')
                plt.yscale('log')
                plt.xlim(PLOT_L_MIN, PLOT_L_MAX)
                plt.title(p)
                plt.ylim(1e34, 1e39)

                plt.savefig(filename + '.png', bbox_inches='tight')
                plt.close()

    print("SED model components:", ' '.join(sed.contribution_names))
    print("SED info available:", ' '.join(sed.info.keys()))
    print("Selected for analysis:", len(analysed_variables), analysed_variables)



def generate_fluxes(Ngen=100000):
    """Generate random SEDs from the configuration and save the fluxes."""
    # redshifts between 0 and 7, with a wide peak around 1-3.
    rv_redshift = scipy.stats.beta(2, 5, scale=7)
    prior_transform = make_prior_transform(rv_redshift, num_redshift_points=400)
    cache_filters = {}

    fluxdata = np.empty((Ngen, len(param_names + analysed_variables) + len(filters))) * np.nan
    u = np.random.uniform(size=len(param_names))
    assert module_list[-1] == 'redshifting'
    with FastAttenuation():
        for i in tqdm.trange(Ngen):
            while True:
                u[i % len(param_names)] = np.random.uniform()
                # last four are always updated, because they are not cached
                u[-4:] = np.random.uniform(size=len(u[-4:]))
                parameters = prior_transform(u)
                stellar_mass = 10**parameters[-4]
                L_AGN = 10**parameters[-3]
                redshift = parameters[-2]
                parameter_list_here = make_parameter_list(parameters)
                parameter_list_here[-1] = dict(redshift=redshift)

                sed, _, agn_sed = scale_sed_components(module_list, parameter_list_here, stellar_mass, L_AGN)
                sed.cache_filters = cache_filters

                model_fluxes_full, model_variables = compute_model_fluxes(sed, filters)
                if (model_fluxes_full >= 0).all():
                    fluxdata[i][:len(param_names)] = parameters
                    fluxdata[i][len(param_names):len(param_names + analysed_variables)] = model_variables
                    fluxdata[i][len(param_names + analysed_variables):] = model_fluxes_full
                    break
            assert np.isfinite(fluxdata[i]).all(), fluxdata[i]

    # save as fits file
    tout = Table(data=fluxdata, names=param_names + analysed_variables + filters)
    tout.write('model_fluxes.fits', overwrite=True)


def chi2_with_norm(model_fluxes, agn_model_fluxes, obs_fluxes, obs_errors, obs_filter_wavelength, redshift, sys_error, NEV, transmitted_fraction, exponent=2):
    """Likelihood considering all variance contributions.

    Parameters
    ----------
    model_fluxes: array
        list of SED fluxes
    agn_model_fluxes: array
        list of SED fluxes from the AGN components alone
    obs_fluxes: array
        list of measured fluxes
    obs_errors: array
        uncertainties for obs_fluxes
    obs_filter_wavelength: array
        observed-frame wavelength of the filters
    redshift: float
        Redshift
    sys_error: float
        fractional systematic error to apply
    NEV: float
        normalised excess variance to consider on AGN components due to variability
    transmitted_fraction: array
        for each filter, the fraction of flux transmitted through any attenuation.
        This is the ratio of flux to the flux if there was no attenuation.
    exponent: float
        2 for Gaussian statistics (L2), 1 for exponential statistics (L1) being more permissive to outliers

    Returns
    -------
    norm: float
        logarithm of Gaussian likelihood normalisation factor
    chi2: float
        "chi-square", i.e., the -0.5 times the Gaussian likelihood exponential factor
    total_variance: array
        total variance from all contributions (data, variability, model uncertainties)
    """
    # Some observations may not have flux values in some filter(s), but
    # they can have upper limit(s). To process upper limits, the user
    # is asked to put the upper limit as flux value and an error value with
    # (obs_errors>=-9990. and obs_errors<0.).
    # Next, the user has two options:
    # 1) s/he puts True in the boolean lim_flag
    # and the limits are processed as upper limits below.
    # 2) s/he puts False in the boolean lim_flag
    # and the limits are processed as no-data below.

    # χ² of the comparison of each model to each observation.
    # This mask selects the filter(s) for which measured fluxes are given
    # i.e., when (obs_flux is >=0. and obs_errors>=0.) and lim_flag=True
    mask_data = np.logical_and(obs_fluxes > TOLERANCE,
                               obs_errors > TOLERANCE)
    # This mask selects the filter(s) for which upper limits are given
    # i.e., when (obs_flux is >=0. (and obs_errors>=-9990., obs_errors<0.))
    # and lim_flag=True
    mask_lim = np.logical_and(obs_errors >= -9990., obs_errors < TOLERANCE)

    # (1) variance from observation, according to the reported errors
    obs_variance = obs_errors[mask_data]**2

    # (2) variance from year-to-year variability (Simm+ paper)
    if variability_uncertainty:
        var_variance = NEV * agn_model_fluxes**2
    else:
        var_variance = 0.0

    # (3) variance from model systematic uncertainties
    sys_variance = (sys_error * model_fluxes)**2

    if with_attenuation_model_uncertainty:
        # (3b) attenuation model error
        # map transmitted fraction (from 1..0.1..0.01 [0..1..2]) to standard deviation (0..0.01..0.01)
        transmitted_fraction[transmitted_fraction < 1e-4] = 1e-4
        transmitted_fraction[transmitted_fraction > 1.0] = 1.0
        neg_log_transmitted = -np.log10(transmitted_fraction + 1e-4)
        # powerlaw increase from 0->1e-4 to 1->0.01  (2->1)
        log_uncertainty_fraction = -4 + 2 * neg_log_transmitted
        # but threshold at 1%
        log_uncertainty_fraction[log_uncertainty_fraction > -2] = -2
        # this is the uncertainty relative to the unattenuated spectrum
        # since we apply to total fluxes, we need to correct it upwards.
        attenuation_uncertainty = 10**log_uncertainty_fraction / transmitted_fraction
        # print("attenuation model uncertainties:", transmitted_fraction, 10**log_uncertainty_fraction, attenuation_uncertainty)
        # apply as a fraction of total model fluxes
        sys_variance += (attenuation_uncertainty * model_fluxes)**2
    del obs_filter_wavelength
    del redshift

    # combined variance in each filter
    total_variance = obs_variance + sys_variance + var_variance

    # compute chi^2 and the Gaussian likelihood normalisation
    chi2_ = np.sum(
        ((obs_fluxes[mask_data]-model_fluxes[mask_data])**2 / total_variance)**(exponent/2.0))
    norm = 0.5 * np.log(2 * np.pi * total_variance**(exponent/2.0)).sum()

    if mask_lim.any():
        uplim_errors = (-obs_errors[mask_lim])**2 + sys_variance + var_variance
        chi2_ += -2. * log(
                np.sqrt(np.pi/2.)*(-obs_errors[mask_lim])*(
                    1.+erf(
                        (obs_fluxes[mask_lim]-model_fluxes[mask_lim]) /
                        (np.sqrt(2)*(uplim_errors))))).sum()
    return norm, chi2_, total_variance


class ModelLikelihood(object):
    """
    Likelihood function, which also knows about the observational data.

    Parameters
    ----------
    wobs: list of bool
        which filters are active
    obs_fluxes: list
        observed flux values
    obs_errors: list
        uncertainties for obs_fluxes
    obs_filter_wavelength: list
        observed_frame wavelength of each filter
    """

    def __init__(self, wobs, obs_fluxes, obs_errors, obs_filter_wavelength):
        self.cache_filters = {}
        self.last_parameters = None
        self.wobs = wobs
        self.obs_fluxes = obs_fluxes
        self.obs_errors = obs_errors
        self.obs_filter_wavelength = obs_filter_wavelength
        self.last_loglikelihood = None

    def __call__(self, parameters):
        """Fitting likelihood function"""

        # if we are called with the same values again, return what we just computed
        # this can happen because the parameters are binned
        if self.last_parameters is not None and np.all(self.last_parameters == parameters):
            return self.last_loglikelihood

        # get the normalisation parameters
        stellar_mass = 10**parameters[-4]
        L_AGN = 10**parameters[-3]
        redshift = parameters[-2]
        sys_error = parameters[-1]
        parameter_list_here = make_parameter_list(parameters)
        assert module_list[-1] == 'redshifting'
        parameter_list_here[-1] = dict(redshift=redshift)

        # compute SED
        sed, gal_sed, agn_sed = scale_sed_components(module_list, parameter_list_here, stellar_mass, L_AGN)
        sed.cache_filters = self.cache_filters
        agn_sed.cache_filters = self.cache_filters

        model_fluxes_full, model_variables = compute_model_fluxes(sed, filters)
        sfr = model_variables[analysed_variables.index('sfh.sfr100Myrs')]
        if not 0 <= sfr <= args.sfr_max:
            # excluded by exceeding age of Universe
            # assign lower number for those further away from the constraints
            logl = -1e20 * (np.log10(stellar_mass) + abs(sfr) + max(0, sed.info['sfh.age'] - sed.info['universe.age']))
            #print("violation", (0, sfr, args.sfr_max), (sed.info['sfh.age'], sed.info['universe.age']), "-->", logl)
            self.last_parameters = parameters
            self.last_loglikelihood = logl
            return logl

        for module_name, module_parameters in zip(module_list[cache_depth:], parameter_list_here[cache_depth:]):
            module_instance = creation_modules.get_module(module_name, **module_parameters)
            module_instance.process(agn_sed)

        agn_model_fluxes_full, _ = compute_model_fluxes(agn_sed, filters)
        agn_model_fluxes = agn_model_fluxes_full[self.wobs]
        model_fluxes = model_fluxes_full[self.wobs]

        # get fraction of non-attenuated flux at the filters
        filter_wl_indices = np.searchsorted(sed.wavelength_grid, filters_wl_orig[self.wobs])
        filter_contrib = sed.luminosities[:, filter_wl_indices]
        filter_pos_contrib = np.where(filter_contrib > 0, filter_contrib, 0).sum(axis=0)
        transmitted_fraction = filter_contrib.sum(axis=0) / filter_pos_contrib

        # compute likelihood:
        norm, chi2_, _ = chi2_with_norm(
            model_fluxes, agn_model_fluxes, self.obs_fluxes, self.obs_errors,
            self.obs_filter_wavelength, redshift, sys_error, NEV=sed.info['agn.NEV'],
            exponent=exponent, transmitted_fraction=transmitted_fraction)

        logl = -0.5 * chi2_ - norm
        # for a Gaussian(0,1) prior on log10(SFR), add
        # logl += -0.5 * (np.log10(sfr + 1e-4))**2
        self.last_parameters = parameters
        self.last_loglikelihood = logl
        return logl


def analyse_obs_wrapper(args):
    """Wrapper catching crashes to continue with the next source.

    When analysing large samples with large machines, it can be annoying if
    an analysis fails due to file locking or files being deleted etc.
    This allows continuing the run with the next source,
    and later reprocessing the entire sample.
    """
    samplername, obs, plot = args
    try:
        return analyse_obs(samplername, obs, plot=plot)
    except OSError as e:
        print("skipping '%s', probably analysed on another machine. error was: '%s'" % (obs['id'], e))
        return obs['id'], None, None
    except RuntimeError as e:
        print("skipping '%s', probably analysed on another machine. error was: '%s'" % (obs['id'], e))
        return obs['id'], None, None
    except BlockingIOError as e:
        print("skipping '%s', probably analysed on another machine. error was: '%s'" % (obs['id'], e))
        return obs['id'], None, None
    except np.linalg.LinAlgError as e:
        print("skipping '%s', probably not enough data points. error was: '%s'" % (obs['id'], e))
        return obs['id'], None, None


def analyse_obs(samplername, obs, plot=True):
    """Source fitting.

    Parameters
    ----------
    obs: dict
        Observation table row (id, redshift, etc)
    plot: bool
        whether to produce plots
    samplername: str
        which fitting algorithm to run.
        MCMC, optimization, MLFriends were supported in a previous version
        but are inefficient or get stuck.
        samplername='nested-slice' is the only currently supported and works well.
        It runs nested sampling with MLFriends initially, then switches over to
        slice sampling.

    Returns
    -------
    id: str
        source id
    results: array
        list of fitting result values. May be None if analysis failed.
    results_string: str
        same as results, but converted to a string. Can be present even if results is None.
    """
    assert samplername == 'nested-slice'

    redshift_mean = obs['redshift']
    num_redshift_points = 40
    if samplername.startswith('nested') and ('redshift_err' not in obs.colnames or 0<=obs['redshift_err']<=0.001):
        rv_redshift = DeltaDist(redshift_mean)
        active_param_names = param_names[:-1]
        derived_param_names = [param_names[-1]]
    else:
        if 'redshift_err' in obs.colnames:
            redshift_err = obs['redshift_err']
        else:
            # put a 1% error on 1+z, at least
            redshift_err = 0.01 * redshift_mean
        if redshift_err < 0:
            num_redshift_points = 200
            print("unknown-z mode: flat redshift prior from 0 to 6")
            # flat redshift distribution, photo-z mode
            rv_redshift = scipy.stats.uniform(0.001, 6)
        else:
            rng = np.random.default_rng(42)
            redshift_samples = rng.normal(redshift_mean, redshift_err, size=1000)
            redshift_samples = redshift_samples[redshift_samples>0]
            redshift_shape, _, redshift_scale = scipy.stats.weibull_min.fit(
                redshift_samples, floc=0,
            )
            rv_redshift = scipy.stats.weibull_min(redshift_shape, scale=redshift_scale)
            print("photo-z mode: redshift prior: ", redshift_shape, redshift_scale)
        active_param_names = param_names
        derived_param_names = []

    print()
    print("="*80)
    print()
    print("Source:", obs['id'], "Redshift:", rv_redshift.mean(), rv_redshift.std())
    print()
    if 'FAGN' in obs.keys() and 'FAGN_errlo' in obs.keys() and 'FAGN_errhi' in obs.keys():
        Finfo = (obs['FAGN'], obs['FAGN_errlo'], obs['FAGN_errhi'])
        print("Using AGN flux constraint:", Finfo)
    elif 'FAGN' in obs.keys() and 'FAGN_err' in obs.keys():
        Finfo = (obs['FAGN'], obs['FAGN_err'], obs['FAGN_err'])
        print("Using AGN flux constraint:", Finfo)
    else:
        Finfo = None

    prior_transform = make_prior_transform(rv_redshift, Finfo=Finfo, num_redshift_points=num_redshift_points)

    prior_samples = np.asarray([prior_transform(u) for u in np.random.uniform(size=(10000, len(active_param_names)))])
    assert np.isfinite(prior_samples[0]).all(), (prior_samples[0])
    assert np.isfinite(prior_samples).all(), (
        np.where(~np.isfinite(prior_samples).all(axis=0)),
        np.where(~np.isfinite(prior_samples).all(axis=1)),
        prior_samples[~np.isfinite(prior_samples)])

    # select the filters from the list of active filters

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        obs_fluxes_full = np.array([obs[name] for name in filters])
        obs_errors_full = np.array([obs[name + "_err"] for name in filters])

    wobs = np.where(obs_fluxes_full > TOLERANCE)
    obs_fluxes = obs_fluxes_full[wobs]
    obs_errors = obs_errors_full[wobs]
    obs_filter_wavelength = filters_wl_orig[wobs]

    loglikelihood = ModelLikelihood(wobs, obs_fluxes, obs_errors, obs_filter_wavelength)

    outdir = "grahsp_%s_var%s%s%d" % (
        str(obs['id']).strip(),
        'V' if variability_uncertainty else '',
        'A' if with_attenuation_model_uncertainty else '',
        -int(np.log10(systematics_width)),
    )
    if args.mass_max != 15:
        outdir += "_maxgal%d" % args.mass_max

    replot = not os.path.exists(outdir + '/analysis_results.txt')
    results = None
    with FastAttenuation():
        try:
            sampler = ReactiveNestedSampler(
                active_param_names, loglikelihood, prior_transform,
                log_dir=outdir, resume='resume', derived_param_names=derived_param_names)
        except Exception as e:
            print("WARNING: could not resume because of %s. overwriting." % e)
            os.unlink(outdir + '/results/points.hdf5')
            # previous results are invalid, so start from scratch
            replot = True
            sampler = ReactiveNestedSampler(
                active_param_names, loglikelihood, prior_transform,
                log_dir=outdir, resume='overwrite', derived_param_names=derived_param_names)
        print("  running without step sampler ...")
        sampler_args = dict(
            frac_remain=0.5, max_num_improvement_loops=0, min_num_live_points=50,
            dlogz=10, min_ess=100, cluster_num_live_points=0, viz_callback=None
        )
        sampler.run(max_ncalls=10000, **sampler_args)
        print("  running with step sampler ...")
        sampler.stepsampler = ultranest.stepsampler.SliceSampler(
            nsteps=20, generate_direction=ultranest.stepsampler.generate_mixture_random_direction)
        sampler.run(**sampler_args)
        sampler.print_results()
    if plot:
        results = plot_results(
            sampler, prior_samples, obs, obs_fluxes, obs_errors, wobs,
            loglikelihood.cache_filters, replot=replot)
    sampler.pointstore.close()
    if results is not None:
        names, means, stds, los, his, medians, lmeans = results
        with open(outdir + '/analysis_results.txt', 'w') as fout:
            fout.write("%s" % obs['id'])
            for name, mean, std, lo, hi, med, lmean in zip(names, means, stds, los, his, medians, lmeans):
                fout.write("\t%g\t%g\t%g\t%g\t%g\t%g" % (mean, std, lo, hi, med, lmean))
            fout.write('\n')
    try:
        results_string = open(outdir + '/analysis_results.txt', 'r').read()
    except IOError:
        results_string = None

    print("  results stored in %s" % outdir)
    return obs['id'], results, results_string


def main():
    """Script entry point, calls function depending on the mode."""
    if args.action == 'generate-from-prior':
        generate_fluxes()
    elif args.action == 'list-filters':
        list_filters()
    elif args.action == 'plot-model':
        plot_model()
    else:
        plot = args.plot
        # Read the observation table and complete it by adding error where
        # none is provided and by adding the systematic deviation.
        obs_table = complete_obs_table(read_table(data_file), column_list,
                                       filters, TOLERANCE, lim_flag)

        # pick observations to analyse in this process
        obs_table_here = obs_table[args.offset::args.every]
        indices = np.arange(len(obs_table_here))
        if args.randomize:
            np.random.shuffle(indices)
        fout = None
        # analyse observations in parallel
        if args.cores == 1:
            # to preserve traceback for debugging run in here
            allresults = (analyse_obs_wrapper((args.sampler, obs_table_here[i], plot)) for i in indices)
        else:
            with mp_ctx.Pool(args.cores, maxtasksperchild=3) as pool:
                # farm out to process pool
                allresults = pool.imap_unordered(
                    analyse_obs_wrapper,
                    ((args.sampler, obs_table_here[i], plot) for i in indices))
        for id, result, results_string in allresults:
            if results_string is None:
                print("no result to store for", id, ". Delete plots, otherwise results will not be reanalysed.")
                continue
            derived_names = analysed_variables + ['NEV', 'LbolBBB', 'LbolTOR', 'chi2']
            names = param_names + derived_names
            names += ['s_' + n for n in derived_names if 'sfh.sfr' in n or 'agn.lum' in n or 'Lbol' in n]
            names += ['totalflux_' + filtername for filtername in filters]
            names += ['AGNflux_' + filtername for filtername in filters]
            names += ['GALflux_' + filtername for filtername in filters]
            if fout is None:
                fout = open(data_file + '_analysis_results.txt', 'w')
                fout.write('# id')
                for name in names:
                    fout.write('\t%s_mean\t%s_std\t%s_lo\t%s_hi\t%s_med\t%s_lmean' % (name, name, name, name, name, name))
                fout.write('\n')
            fout.write(results_string)
            fout.flush()

        print("analying %d observations done." % len(obs_table_here))


if __name__ == '__main__':
    main()

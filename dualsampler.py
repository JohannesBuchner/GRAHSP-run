"""
GRAHSP (Grasping reliably the AGN host stellar population)

A broad-band SED fitting tool (X-ray to mid-infrared) for investigating host galaxies of
active galactic nuclei.

Features:

- Flexible empirical AGN model to avoid modelling bias
- broad and narrow line emission from AGN, in addition to accretion disk and obscurer.
- allows obscurer diversity (hotter and colder dust) and 12µm Si in emission or absorption
- allows different extinction for AGN and host, with the more appropriate Prevot law.
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
from math import erf

import scipy.stats
from scipy.constants import c
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ultranest import ReactiveNestedSampler
from ultranest.plot import PredictionBand
import tqdm
import joblib
import getdist, getdist.plots, getdist.chains
getdist.chains.print_load_details = False
import logging

class DeltaDist(object):
    """Dirac Delta distribution."""
    def __init__(self, value):
        self.value = value
    def ppf(self, u):
        return self.value
    def mean(self):
        return self.value
    def std(self):
        return 0


class FastExtinction(object):
    """Context within which the extinction law reduces computation.

    Per-filter extinctions, and per-component extinction are skipped.
    """
    def __enter__(self):
        ExtinctionLaw.store_filter_attenuation = False
        ExtinctionLaw.store_component_attenuation = False
    def __exit__(self, type, value, traceback):
        ExtinctionLaw.store_filter_attenuation = True
        ExtinctionLaw.store_component_attenuation = True


class HelpfulParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)


parser = HelpfulParser(description=__doc__,
	epilog="""Johannes Buchner (C) 2013-2022 <johannes.buchner.acad@gmx.com>""",
	formatter_class=argparse.RawDescriptionHelpFormatter)


parser.add_argument('--offset', type=int, default=0,
	help='row in the input file to begin processing')

parser.add_argument('--every', type=int, default=1,
	help='stride in the input file to process (every nth row)')

parser.add_argument('--cores', type=int, default=-1,
	help='number of processes to parallelise for (see joblib.Parallel)')

parser.add_argument('--plot', action='store_true',
	help='also make plots of the SED and parameter constraints')
parser.add_argument('--mass-max', type=float, default=15,
	help='Maximum stellar mass.')
parser.add_argument('--randomize', action='store_true',
	help='Randomize order in which to analyse observations.')

parser.add_argument('action', type=str, default='analyse', choices=('analyse', 'generate-from-prior', 'plot-model', 'list-filters'),
	help='''Mode. Generate a file with fluxes from randomly drawn model instances, plot model SED variations, list the available photometric filters, or (default) analyse a photometry file.''')

parser.add_argument('--sampler', type=str, default='nested-slice', choices=('nested', 'mcmc', 'laplace', 'nested-slice', 'noop'),
	help='Parameter space sampling algorithm to use. Nested-slice is recommended.')


args = parser.parse_args()

print("loading cigale ...")
from pcigale.session.configuration import Configuration
from pcigale.analysis_modules import get_module as get_analysis_module
from pcigale.utils import read_table
from pcigale.analysis_modules import complete_obs_table
from pcigale.warehouse import SedWarehouse
from pcigale import creation_modules
from pcigale.analysis_modules.pdf_analysis import TOLERANCE
from pcigale.data import Database
gbl_warehouse = SedWarehouse()
from pcigale.creation_modules.extinction import ExtinctionLaw

print("parsing pcigale.ini...")
config = Configuration("pcigale.ini")

data_file = config.configuration['data_file']
column_list = config.configuration['column_list']
module_list = config.configuration['creation_modules']
statistics_config = config.config['statistics']
# n_cores = int(config.configuration['cores'])
n_cores = args.cores
parameter_list = config.configuration['creation_modules_params']
cache_depth = module_list.index('extinction')
if module_list[cache_depth - 1] == 'activatepl':
    cache_depth -= 1
cache_max = int(os.environ.get('CACHE_MAX', '10000'))
chunk_size = int(os.environ.get('CHUNKSIZE', '20'))
cache_print = os.environ.get('CACHE_VERBOSE', '0') == '1'
if cache_print:
    print("Caching modules:", module_list[:cache_depth], "with %d entries" % cache_max)
analysis_module = get_analysis_module(config.configuration[
    'analysis_method'])
analysis_module_params = config.configuration['analysis_method_params']

analysed_variables = analysis_module_params["analysed_variables"]
n_variables = len(analysed_variables)
lim_flag = analysis_module_params["lim_flag"].lower() == "true"
mock_flag = analysis_module_params["mock_flag"].lower() == "true"

exponent = int(statistics_config.get('exponent', '2'))
with_uv_model_uncertainty = statistics_config.get('uv_model_uncertainty', 'false').lower() == 'true'
variability_uncertainty = statistics_config.get('variability_uncertainty', 'true').lower() == 'true'
systematics_width = float(statistics_config.get('systematics_width', '0.01'))


filters = [name for name in column_list if not name.endswith('_err')]
with Database() as base:
    filters_wl_orig = np.array([base.get_filter(name.rstrip('_')).effective_wavelength for name in filters])
n_filters = len(filters)


param_names = []
is_log_param = []
print()
print("Parameters")
print("----------")
for module_name, module_parameters in zip(module_list, parameter_list):
    print("  [%s]" % module_name)
    for k, v in module_parameters.items():
        if len(v) > 1:
            if min(v) > 0 and max(v) > 0 and max(v) / min(v) > 40:
                is_log_param.append(True)
                param_names.append("log(%s.%s)" % (module_name, k))
            else:
                is_log_param.append(False)
                param_names.append("%s.%s" % (module_name, k))
            print("    %20s : %s" % (k, v), '(log-uniform)' if is_log_param[-1] else '(uniform)')
        else:
            print("    %20s = %s" % (k, v[0]))
        del k, v

    del module_name, module_parameters
print()
param_names.append("log(stellar_mass)")
param_names.append("log(L_AGN)")
param_names.append("redshift")
param_names.append("systematics")
rv_systematics = scipy.stats.halfcauchy(scale=systematics_width)

print("Statistics")
print("----------")
print(" Exponent: %d (%s)" % (exponent, {1:'L1',2:'Gaussian'}[exponent]))
print(" model uncertainty: ")
print("    white: half-cauchy width: %s" % systematics_width)
print("    wavelength-dependent: UV: %s" % with_uv_model_uncertainty)
print("    variability: %s" % variability_uncertainty)
print()

def make_parameter_list(parameters):
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

def get_model_fluxes(sed, filters):
    if 'sfh.age' in sed.info and sed.info['sfh.age'] > sed.info['universe.age']:
        model_fluxes = -99. * np.ones(len(filters))
        model_variables = -99. * np.ones(len(analysed_variables))
    else:
        model_fluxes = np.array([sed.compute_fnu(filter_.rstrip('_')) for filter_ in filters])
        model_variables = np.array([sed.info[name] for name in analysed_variables])

    return model_fluxes, model_variables



def scale_sed_components(module_list, parameter_list_here, stellar_mass, L_AGN):
    # get sed for galaxy
    parameter_list_gal = []
    parameter_list_agn = []
    for module_name, module_parameters_available, module_parameters_selected in zip(module_list, parameter_list, parameter_list_here):
        parameter_list_gal_here = {}
        parameter_list_agn_here = {}
        for k in sorted(module_parameters_available.keys()):
            selected_value = module_parameters_selected[k]
            mock_value = module_parameters_available[k][0]
            if k in ('redshift', 'AGNtype', 'E(B-V)', 'E(B-V)-AGN'):
                # use the true value in both cases, because:
                # 1) redshift applies to both
                # 2) extinction module is applied after both and not cached
                # 3) AGNtype varies the number of parameters
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

    if len(gbl_warehouse.storage.dictionary) > cache_max:
        if cache_print: print("clearing cache:", len(gbl_warehouse.storage.dictionary))
        gbl_warehouse.partial_clear_cache(cache_depth)
        if cache_print: print("cleared  cache:", len(gbl_warehouse.storage.dictionary))
    sed = gbl_warehouse.get_sed(module_list[:cache_depth], parameter_list_gal[:cache_depth])
    agn_sed = gbl_warehouse.get_sed(module_list[:cache_depth], parameter_list_agn[:cache_depth])
    assert sed.contribution_names == agn_sed.contribution_names, (sed.contribution_names, agn_sed.contribution_names)

    agn_mask = np.array(['activate' in name for name in sed.contribution_names])
    assert np.isfinite(sed.luminosities).all(), sed.luminosities
    assert np.isfinite(sed.luminosity).all(), sed.luminosity

    # remember igm attenuation
    # igm = sed.luminosities[-1] / (sed.luminosity + 1e-100)
    # assert np.isfinite(igm).all(), igm
    scaled_sed = sed.copy()
    # scale the AGN and galactic components as needed
    scaled_sed.luminosities[~agn_mask] *= stellar_mass
    scaled_sed.info.update({k:v * stellar_mass for k, v in sed.info.items() if k in sed.mass_proportional_info})
    # convert from erg/s/A at 5100A to erg/s with 5100
    # convert from erg/s to W, the luminosity unit of cigale, with 1e7
    scaled_sed.luminosities[agn_mask] = agn_sed.luminosities[agn_mask] * L_AGN / 1e7 / 5100
    # copy over AGN meta data
    scaled_sed.info.update({k:v * (L_AGN if 'agn.lum' in k else 1) for k, v in agn_sed.info.items() if 'activate' in k or 'agn' in k})
    scaled_sed.luminosity = scaled_sed.luminosities.sum(0)

    # apply the remaining modules
    for module_name, module_parameters in zip(module_list[cache_depth:], parameter_list_gal[cache_depth:]):
        module_instance = creation_modules.get_module(module_name, **module_parameters)
        module_instance.process(scaled_sed)

    #print("IGM:", igm.min(), igm.max())
    assert np.isfinite(scaled_sed.luminosities[:-1].sum(0)).all(), scaled_sed.luminosities[:-1].sum(0)
    # recompute IGM attenuation
    #scaled_sed.luminosities[-1] = igm * scaled_sed.luminosities[:-1].sum(0)
    assert np.isfinite(scaled_sed.luminosities).all(), scaled_sed.luminosities
    
    # update total luminosities
    assert np.isfinite(scaled_sed.luminosity).all(), scaled_sed.luminosity
    return scaled_sed, sed, agn_sed

# Wavelength limits (restframe) when plotting the best SED.
PLOT_L_MIN = 0.1
PLOT_L_MAX = 50

def with_attenuation(keys):
    return keys + ['attenuation.' + key for key in keys]

def plot_posteriors(filename, prior_samples, param_names, samples):
    print("plotting posteriors ...")
    plt.figure(figsize=(12, 12))
    for i, (param_name, samples) in enumerate(zip(param_names, samples.transpose())):
        plt.subplot(4, len(param_names) // 4 + 1, i + 1)
        bins = np.unique(list(set(prior_samples[:,i]).union(set(samples))))
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
            plt.hist(prior_samples[:,i], histtype='step', 
                density=True, bins=bins, color='gray', ls='-')
        plt.xlim(xlo, xhi)
        plt.yticks([])
        plt.xlabel(param_names[i])
    
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

plot_elements = [
    dict(keys=with_attenuation(['stellar.young', 'stellar.old']),
         label="Stellar attenuated", color='orange', marker=None, linestyle='-',),
    #dict(keys=['stellar.young', 'stellar.old'],
    #     label="Stellar unattenuated", color='b', marker=None, nonposy='clip', linestyle='--', linewidth=0.5),
    dict(keys=with_attenuation(['nebular.lines_young', 'nebular.lines_old', 'nebular.continuum_young', 'nebular.continuum_old']),
         label="Nebular emission", color='y', marker=None, linewidth=.5),
    dict(keys=with_attenuation(['agn.activate_Disk']),
         label="AGN disk", color=[0.90, 0.90, 0.72], marker=None, linestyle='-', linewidth=1.5),
    dict(keys=with_attenuation(['agn.activate_Torus']),
         label="AGN torus", color=[0.90, 0.77, 0.42], marker=None, linestyle='-', linewidth=1.5),
    dict(keys=with_attenuation(['agn.activate_EmLines_BL', 'agn.activate_EmLines_NL', 'agn.activate_FeLines', 'agn.activate_EmLines_LINER']),
         label="AGN lines", color=[0.90, 0.50, 0.21], marker=None, linestyle='-', linewidth=0.5),
    dict(keys=['dust'], label="Dust", color='darkred', marker=None, linestyle='-', linewidth=0.5),
    #dict(keys=['F_lambda_total'],
    #     label="Model spectrum", color='k', marker=None, nonposy='clip', linestyle='-', linewidth=1.5, alpha=0.7),
]

def plot_results(sampler, prior_samples, obs, obs_fluxes, obs_errors, wobs, cache_filters, replot):
    # only allow the main process to plot
    if not sampler.log:
        return

    results = sampler.results
    plot_dir = sampler.logs['plots']
    assert np.isfinite(results['samples']).all()
    
    if not replot and all((os.path.exists('%s/%s.pdf' % (plot_dir, f)) for f in ('posteriors', 'derived', 'sed_mJy', 'sed_lum', 'corner'))):
        print("not replotting.")
        return
    plot_posteriors('%s/posteriors.pdf' % plot_dir, prior_samples, param_names, results['samples'])
    print("making corner plot ...")
    smooth_samples = sampler.results['samples'].copy()
    #smooth_prior_samples = prior_samples.copy()
    for i in range(len(param_names)):
        bins = np.unique(prior_samples[:,i]).tolist()
        if len(bins) < 40:
            if len(bins) == 1:
                db = 1
            else:
                db = (bins[-1] - bins[-2])
            for lo, hi in zip(bins, bins[1:] + [bins[-1] + db]):
                mask = smooth_samples[:,i] == lo
                smooth_samples[mask,i] = np.random.uniform(lo, hi, size=mask.sum())
                mask2 = prior_samples[:,i] == lo
                #smooth_prior_samples[mask2,i] = np.random.uniform(lo, hi, size=mask2.sum())
    samples = getdist.MCSamples(
        samples=smooth_samples, names=param_names, sampler='nested',
        settings=dict(smooth_scale_2D=0.3, smooth_scale_1D=0.3))
    #samples2 = getdist.MCSamples(
    #    samples=smooth_prior_samples, names=param_names, sampler='nested',
    #    settings=dict(smooth_scale_2D=0.3, smooth_scale_1D=0.3))
    g = getdist.plots.get_subplot_plotter()
    g.triangle_plot([samples])
    #corner.corner(smooth_samples, labels=param_names, show_titles=True, bins=10)
    for handler in logging.root.handlers[:]: 
         logging.root.removeHandler(handler)
    plt.savefig('%s/corner.pdf' % plot_dir)
    plt.close()

    print("making SED instances ...")
    bands = {'lum':{}, 'mJy':{}}
    
    z = obs['redshift']
    # chi2_best = -2 * sampler.results['weighted_samples']['logl'].max()
    # chi2_reduced = chi2_best / wobs.sum()
    chi2_best = 1e300
    
    posteriors_names = analysed_variables + ['NEV', 'Lbol', 'chi2']
    posteriors = []
    all_mod_fluxes = []
    obs_filter_wavelength = filters_wl_orig[wobs]
    
    for parameters in tqdm.tqdm(sampler.results['samples'][:50,:]):
        stellar_mass = 10**parameters[-4]
        L_AGN = 10**parameters[-3]
        redshift = parameters[-2]
        sys_error = parameters[-1]
        parameter_list_here = make_parameter_list(parameters)
        parameter_list_here[-1] = dict(redshift=redshift)

        sed, gal_sed, agn_sed = scale_sed_components(module_list, parameter_list_here, stellar_mass, L_AGN)
        sed.cache_filters = cache_filters
        model_fluxes_full, model_variables = get_model_fluxes(sed, filters)

        for module_name, module_parameters in zip(module_list[cache_depth:], parameter_list_here[cache_depth:]):
            module_instance = creation_modules.get_module(module_name, **module_parameters)
            module_instance.process(agn_sed)

        agn_model_fluxes_full, _ = get_model_fluxes(agn_sed, filters)
        agn_model_fluxes = agn_model_fluxes_full[wobs]

        NEV, Lbol = compute_NEV(L_AGN)

        mod_fluxes = model_fluxes_full[wobs]
        all_mod_fluxes.append(model_fluxes_full)

        _, chi2_0 = chi2_with_norm(mod_fluxes, agn_model_fluxes*0, obs_fluxes, obs_errors, obs_filter_wavelength * np.inf, redshift, sys_error*0+0.1, NEV, exponent=exponent)
        chi2_best = min(chi2_0, chi2_best)
        norm, chi2_ = chi2_with_norm(mod_fluxes, agn_model_fluxes, obs_fluxes, obs_errors, obs_filter_wavelength, redshift, sys_error, NEV, exponent=exponent)
        posteriors.append(np.concatenate((np.log10(model_variables), [NEV, Lbol, chi2_])))
        
        wavelength_spec = sed.wavelength_grid
        DL = sed.info['universe.luminosity_distance']

        for sed_type in 'mJy', 'lum':
            wavelength_spec2 = wavelength_spec.copy()
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
                    print("  building", sed_type, plot_element['label'])
                    bands[sed_type][j] = PredictionBand(wavelength_spec2)
                pred = sum(sed.get_lumin_contribution(k) * sed_multiplier for k in keys if k in sed.contribution_names)
                assert np.isfinite(pred).all(), pred
                assert bands[sed_type][j].x.shape == pred.shape, (bands[sed_type][j].x.shape, pred.shape)
                # print(plot_element['label'], pred)
                bands[sed_type][j].add(pred)
            if 'total' not in bands[sed_type]:
                bands[sed_type]['total'] = PredictionBand(wavelength_spec2)
            bands[sed_type]['total'].add(sed.luminosity * sed_multiplier)

    print("SED model components:", ' '.join(sed.contribution_names))

    print("SED info available:", ' '.join(sed.info.keys()))
    print("selected:", posteriors_names)
    plot_posteriors('%s/derived.pdf' % plot_dir, np.zeros((0, len(posteriors_names))), posteriors_names, np.array(posteriors))
    np.savetxt('%s/derived.csv' % plot_dir, posteriors, header=','.join(posteriors_names), comments='', delimiter=',')
    
    print("write out flux predictions ...")
    all_mod_fluxes = np.array(all_mod_fluxes)
    mod_fluxes = np.median(all_mod_fluxes, axis=0)[wobs]
    #mod_fluxes = np.median(all_mod_fluxes[:,wobs], axis=0)
    with open('%s/fluxpredict.csv' % plot_dir, 'w') as fflux:
        fflux.write('filtername,flux,flux_err\n')
        for filtername, filterpred in zip(filters, all_mod_fluxes.transpose()):
            fflux.write('%s,%g,%g\n' % (filtername, np.mean(filterpred), np.std(filterpred)))

    print("write out sed ...")
    for sed_type in 'mJy', 'lum':
        header = ['wavelength']
        seddata = [wavelength_spec2]
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

    for sed_type in 'mJy', 'lum':
        filters_wl = filters_wl_orig[wobs] / 1000
        # wsed = np.where((wavelength_spec2 > xmin) & (wavelength_spec2 < xmax))

        figure = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        plt.sca(ax1)
        bands[sed_type]['total'].shade(0.45, color='k', alpha=0.2)
        bands[sed_type]['total'].line(color='k', label='Model spectrum', linewidth=1.5)
        for j, plot_element in enumerate(plot_elements):
            if j in bands[sed_type]:
                print("  plotting", sed_type, plot_element['label'], np.shape(bands[sed_type][j].ys))
                bands[sed_type][j].shade(0.45, color=plot_element['color'], 
                    #label=plot_element['label'] % sed.info, 
                    alpha=0.1)
                line_kwargs = dict(plot_element)
                del line_kwargs['keys']
                bands[sed_type][j].line(**line_kwargs)

        if sed_type == 'lum':
            xmin = PLOT_L_MIN / (1 + z)
            xmax = PLOT_L_MAX / (1 + z)

            filters_wl /= 1. + z
            k_corr_SED = 1e-29 * (4.*np.pi*DL*DL) * c / (filters_wl*1e-9) / 1000
            obs_fluxes = obs_fluxes * k_corr_SED
            obs_fluxes_err = obs_errors * k_corr_SED
            mod_fluxes = mod_fluxes * k_corr_SED
        elif sed_type == 'mJy':
            xmin = PLOT_L_MIN
            xmax = PLOT_L_MAX

            k_corr_SED = 1.
            obs_fluxes_err = obs_errors

        ax1.set_autoscale_on(False)
        ax1.scatter(filters_wl, mod_fluxes, marker='o', color='r', s=8,
                    zorder=3, label="Model fluxes")
        mask_ok = np.logical_and(obs_fluxes > 0., obs_errors > 0.)
        ax1.errorbar(filters_wl[mask_ok], obs_fluxes[mask_ok],
                     yerr=obs_fluxes_err[mask_ok]*3, ls='', marker='s',
                     label='Observed fluxes', markerfacecolor='None',
                     markersize=6, markeredgecolor='b', capsize=2., elinewidth=1)
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
            ax2.set_xlabel("Rest-frame wavelength [$\mu$m]")
            ax1.set_ylabel("Luminosity [W]")
            ax2.set_ylabel("(Obs-Mod)/Obs")
        else:
            ax2.set_xlabel("Observed wavelength [$\mu$m]")
            ax1.set_ylabel("Flux [mJy]")
            ax2.set_ylabel("(Obs-Mod)/Obs")
        ax1.legend(fontsize=6, loc='best', fancybox=True, framealpha=0.5)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels()[1], visible=False)
        figure.suptitle(
            "Best model for %s at z = %.3f. $\chi^2$=%.1f/%d" %
                (obs['id'], obs['redshift'], chi2_best, len(obs_fluxes)))
        figure.savefig("%s/sed_%s.pdf" % (plot_dir, sed_type))
        plt.close(figure)

    return (
        param_names + posteriors_names,
        np.concatenate((results['samples'].mean(axis=0), np.mean(posteriors, axis=0))), 
        np.concatenate((results['samples'].std(axis=0), np.std(posteriors, axis=0))),
        np.concatenate((np.quantile(results['samples'], 0.02275, axis=0), np.quantile(posteriors, 0.02275, axis=0))),
        np.concatenate((np.quantile(results['samples'], 0.97725, axis=0), np.quantile(posteriors, 0.97725, axis=0))),
    )

def make_prior_transform(rv_redshift, Linfo = None):
    redshift_fixed = rv_redshift.std() == 0
    if Linfo is None:
        def L_prior_transform(u):
            # AGN luminosity from 10^38 to 10^50
            return u * 12 + 38
    else:
        LAGN, LAGN_errlo, LAGN_errhi = Linfo
        rv_L_lo = scipy.stats.norm(LAGN, LAGN_errlo)
        rv_L_hi = scipy.stats.norm(LAGN, LAGN_errhi)
        def L_prior_transform(u):
            if u < 0.5:
                return rv_L_lo.ppf(u)
            else:
                return rv_L_hi.ppf(u)
    
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
        
        # AGN luminosity from 10^30 to 10^50
        params[i+1] = L_prior_transform(cube[i+1])
        # * 20 + 30

        # redshift. 
        # params[i+1] = rv_redshift.ppf(cube[i+1])
        # Approximate redshift with points on the CDF
        params[i + 2] = rv_redshift.ppf((1 + np.round(cube[i+2] * 40)) / 42)
        j = i + 2 if redshift_fixed else i + 3
        
        # systematic uncertainty
        params[i + 3] = rv_systematics.ppf(cube[j])
        return params
    return prior_transform

def plot_model():
    umid = np.array([1e-6 if 'E(B-V)' in p else 0.5 for p in param_names])
    redshift = 1.0
    for logL_AGN in [38, 42, 44, 46]:
        L_AGN = 10**logL_AGN
        rv_redshift = scipy.stats.uniform(redshift, redshift+1e-3)
        prior_transform = make_prior_transform(rv_redshift)
        print("reference values:")
        for p, v in zip(param_names, prior_transform(umid)):
            print("   %-20s: %s" % (p, v))
        cache_filters = {}
        
        for i, p in enumerate(param_names):
            if p in ('redshift', 'log(L_AGN)', 'activate.AGNtype', 'systematics'):
                continue
            print("varying", p)

            u = umid.copy()
            for AGNtype in 1, 2, 3:
                last_value = np.nan
                filename = 'modelspectrum_L%d_type%s_%s' % (logL_AGN, AGNtype, p.replace('(','').replace(')',''))
                first_legend = None
                #total_lines = []
                for v in np.linspace(0.001, 0.999, 11):
                    u[i] = v
                    parameters = prior_transform(u)
                    if 'activate.AGNtype' in param_names:
                        parameters[param_names.index('activate.AGNtype')] = AGNtype
                    if parameters[i] == last_value:
                        continue
                    last_value = parameters[i]
                    stellar_mass = 10**parameters[-4]
                    
                    parameter_list_here = make_parameter_list(parameters)
                    assert module_list[-1] == 'redshifting'
                    parameter_list_here[-1] = dict(redshift=redshift)
                    
                    with np.errstate(invalid='ignore'):
                        sed, _, _ = scale_sed_components(module_list, parameter_list_here, stellar_mass, L_AGN)
                    sed.cache_filters = cache_filters

                    _, model_variables = get_model_fluxes(sed, filters)
                    NEV, Lbol = compute_NEV(L_AGN)
                    
                    wavelength_spec = sed.wavelength_grid.copy()
                    wavelength_spec /= 1. + redshift
                    DL = sed.info['universe.luminosity_distance']
                    sed_multiplier = wavelength_spec.copy() * (redshift + 1)
                    assert (sed_multiplier >= 0).all(), (stellar_mass, DL, wavelength_spec)
                    wavelength_spec /= 1000
                    mask = np.logical_and(wavelength_spec >= PLOT_L_MIN, wavelength_spec <= PLOT_L_MAX)
                    alpha = 1 - (v + 0.2) / 1.2

                    outputs = [wavelength_spec]
                    output_labels = ['wavelength']
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
                    
                    #outputs.append(sed.luminosity * sed_multiplier)
                    #output_values.append(parameters)
                    line, = plt.plot(wavelength_spec[mask], (sed.luminosity * sed_multiplier)[mask], '-', color='k', alpha=alpha, label='total')
                    subfilename = filename + '_at%s' % (parameters[i])
                    np.savetxt(subfilename + '.params',
                        [np.concatenate((parameters, model_variables, [NEV, Lbol]))],
                        delimiter=',', header=','.join(param_names + analysed_variables + ['NEV', 'Lbol'])
                    )
                    assert len(outputs)==len(output_labels), (np.shape(outputs), len(output_labels))
                    np.savetxt(subfilename + '.txt', np.transpose(outputs), delimiter=',', header=','.join(output_labels), comments='')
                    
                    if first_legend is None:
                        first_legend = plt.legend(title='Components', framealpha=0.5, loc='upper left')
                plt.gca().add_artist(first_legend)

                plt.xlabel("Wavelength [$\mu$m]")
                plt.ylabel("Luminosity [W/nm]")
                plt.xscale('log')
                plt.yscale('log')
                plt.title(p)
                plt.ylim(1e34, 1e39)
                
                plt.savefig(filename + '.png', bbox_inches='tight')
                plt.close()

def list_filters():
    with Database() as base:
        keys, _ = base.get_filter_list()
        for filter_name in keys:
            f = base.get_filter(filter_name)
            print(f)


def generate_fluxes(Ngen=100000):
    rv_redshift = scipy.stats.uniform(0, 6)
    prior_transform = make_prior_transform(rv_redshift)
    cache_filters = {}

    fluxdata = np.empty((Ngen, len(param_names + analysed_variables) + 2 + len(filters))) * np.nan
    u = np.random.uniform(size=len(param_names))
    for i in tqdm.trange(Ngen):
        u[np.random.randint(len(param_names))] = np.random.uniform()
        parameters = prior_transform(u)
        stellar_mass = 10**parameters[-4]
        L_AGN = 10**parameters[-3]
        redshift = parameters[-2]
        parameter_list_here = make_parameter_list(parameters)
        assert module_list[-1] == 'redshifting'
        parameter_list_here[-1] = dict(redshift=redshift)
        
        NEV, Lbol = compute_NEV(L_AGN)
        sed, _, _ = scale_sed_components(module_list, parameter_list_here, stellar_mass, L_AGN)
        sed.cache_filters = cache_filters

        model_fluxes_full, model_variables = get_model_fluxes(sed, filters)
        fluxdata[i][:len(param_names)] = parameters
        fluxdata[i][len(param_names):len(param_names + analysed_variables)] = model_variables
        fluxdata[i][len(param_names + analysed_variables)] = NEV
        fluxdata[i][len(param_names + analysed_variables) + 1] = Lbol
        fluxdata[i][len(param_names + analysed_variables) + 2:] = model_fluxes_full
        assert np.isfinite(fluxdata[i]).all(), fluxdata[i]

    np.savetxt(
        "model_fluxes.txt.gz", fluxdata, delimiter=',', comments='',
        header=','.join(param_names + analysed_variables + ['NEV', 'Lbol'] + filters))

def chi2_with_norm(model_fluxes, agn_model_fluxes, obs_fluxes, obs_errors, obs_filter_wavelength, redshift, sys_error, NEV, exponent=2):
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
    
    # variance from observation, according to the reported errors
    obs_variance = obs_errors[mask_data]**2
    # variance from year-to-year variability (Simm+ paper)
    if variability_uncertainty:
        var_variance = NEV * agn_model_fluxes**2
    else:
        var_variance = 0.0
    # variance from model systematic uncertainties
    sys_variance = (sys_error * model_fluxes)**2
    if with_uv_model_uncertainty:
        # UV model error
        rest_filter_wavelength = obs_filter_wavelength / (1 + redshift)
        sys_variance[rest_filter_wavelength < 125] = 1e16
        #sys_variance += (4 * (rest_filter_wavelength / 91.0)**-4 * model_fluxes)**2
        # print(np.array(list(zip(filters, obs_filter_wavelength, rest_filter_wavelength, sys_variance))))
    # combined variance in each filter
    total_variance = obs_variance + sys_variance + var_variance

    chi2_ = np.sum(
        ((obs_fluxes[mask_data]-model_fluxes[mask_data])**2 / total_variance)**(exponent/2.0) )
    norm = 0.5 * np.log(2 * np.pi * total_variance**(exponent/2.0)).sum()

    if mask_lim.any():
        uplim_errors = (-obs_errors[mask_lim])**2 + sys_variance + var_variance
        chi2_ += -2. * log(
                np.sqrt(np.pi/2.)*(-obs_errors[mask_lim])*(
                    1.+erf(
                        (obs_fluxes[mask_lim]-model_fluxes[mask_lim]) /
                        (np.sqrt(2)*(uplim_errors))))).sum()
    return norm, chi2_

def compute_NEV(L_AGN):
    # from Netzer+19 bolometric corrections
    Lbol = L_AGN * 40 * (L_AGN / 1e42)**-0.2
    # from Simm+16 Table 3: normalised excess variance as a function of Lbol
    # NEV = min(0.1, 10**(-1.43 - 0.74 * np.log10(Lbol / 1e45)))
    NEV = min(0.1, 10**(-1.43 - 0.74 * np.log10(Lbol / 1e45)))
    # print('L5100=%.1e  Lbol=%.1e  var=%.4f' % (L_AGN, Lbol, NEV))
    return NEV, Lbol

def analyse_obs_wrapper(samplername, obs, plot=True):
    # new source, so start fresh
    gbl_warehouse.partial_clear_cache(0)
    import gc; gc.collect()
    #np.random.seed(1)
    try:
        return analyse_obs(samplername, obs, plot=plot)
    except OSError as e:
        print("skipping, probably analysed on another machine. error was: %s" % e)
        return obs['id'], None, None
    except RuntimeError as e:
        print("skipping, probably analysed on another machine. error was: %s" % e)
        return obs['id'], None, None
    except BlockingIOError as e:
        print("skipping, probably analysed on another machine. error was: %s" % e)
        return obs['id'], None, None

def analyse_obs(samplername, obs, plot=True):
    redshift_mean = obs['redshift']
    if 'redshift_err' not in obs.colnames and samplername.startswith('nested'):
        rv_redshift = DeltaDist(redshift_mean)
        active_param_names = param_names[:-1]
        derived_param_names = [param_names[-1]]
    else:
        if 'redshift_err' in obs.colnames:
            redshift_err = obs['redshift_err']
        else:
            # put a 1% error on 1+z, at least
            redshift_err = 0.01 * redshift_mean
        redshift_samples = np.random.normal(redshift_mean, redshift_err, size=1000)
        redshift_samples = redshift_samples[redshift_samples>0]
        redshift_shape, _, redshift_scale = scipy.stats.weibull_min.fit(
            redshift_samples, floc=0,
        )
        rv_redshift = scipy.stats.weibull_min(redshift_shape, scale=redshift_scale)
        active_param_names = param_names
        derived_param_names = []

    print()
    print("="*80)
    print()
    print("Source:", obs['id'], "Redshift:", rv_redshift.mean(), rv_redshift.std())
    print()
    if 'LAGN' in obs.keys() and 'LAGN_errlo' in obs.keys() and 'LAGN_errhi' in obs.keys():
        Linfo = (obs['LAGN'], obs['LAGN_errlo'], obs['LAGN_errhi']) 
        print("Using X-ray luminosity constraint:", Linfo)
    elif 'LAGN' in obs.keys() and 'LAGN_err' in obs.keys():
        Linfo = (obs['LAGN'], obs['LAGN_err'], obs['LAGN_err'])
        print("Using X-ray luminosity constraint:", Linfo)
    else:
        Linfo = None
    
    prior_transform = make_prior_transform(rv_redshift, Linfo)
    
    prior_samples = np.asarray([prior_transform(u) for u in np.random.uniform(size=(10000, len(active_param_names)))])
    assert np.isfinite(prior_samples).all(), (np.where(~np.isfinite(prior_samples).all(axis=0)), np.where(~np.isfinite(prior_samples).all(axis=1)), prior_samples[~np.isfinite(prior_samples)])

    # select the filters from the list of active filters

    obs_fluxes_full = np.array([obs[name] for name in filters])
    obs_errors_full = np.array([obs[name + "_err"] for name in filters])

    wobs = np.where(obs_fluxes_full > TOLERANCE)
    obs_fluxes = obs_fluxes_full[wobs]
    obs_errors = obs_errors_full[wobs]
    obs_filter_wavelength = filters_wl_orig[wobs]
    #if not np.logical_and(obs_fluxes_full > 0, obs_errors_full > 0).any():
    #    print("ERROR: Source does not have detections, skipping")
    #    return obs['id'], None, None

    cache_filters = {}
    
    def loglikelihood(parameters):
        if loglikelihood.last_parameters is not None and np.all(loglikelihood.last_parameters == parameters):
            return loglikelihood.last_loglikelihood
        stellar_mass = 10**parameters[-4]
        L_AGN = 10**parameters[-3]
        redshift = parameters[-2]
        sys_error = parameters[-1]
        parameter_list_here = make_parameter_list(parameters)
        assert module_list[-1] == 'redshifting'
        parameter_list_here[-1] = dict(redshift=redshift)
        
        sed, gal_sed, agn_sed = scale_sed_components(module_list, parameter_list_here, stellar_mass, L_AGN)
        sed.cache_filters = cache_filters

        model_fluxes_full, model_variables = get_model_fluxes(sed, filters)

        for module_name, module_parameters in zip(module_list[cache_depth:], parameter_list_here[cache_depth:]):
            module_instance = creation_modules.get_module(module_name, **module_parameters)
            module_instance.process(agn_sed)

        agn_model_fluxes_full, _ = get_model_fluxes(agn_sed, filters)
        agn_model_fluxes = agn_model_fluxes_full[wobs]

        model_fluxes = model_fluxes_full[wobs]
        
        NEV, Lbol = compute_NEV(L_AGN)

        # get unobscured luminosities at filter wavelengths
        # agn_sed.luminosities[:-1].sum(0)
        
        norm, chi2_ = chi2_with_norm(model_fluxes, agn_model_fluxes, obs_fluxes, obs_errors, obs_filter_wavelength, redshift, sys_error, NEV, exponent=exponent)
        
        logl = -0.5 * chi2_ - norm
        loglikelihood.last_parameters = parameters
        loglikelihood.last_loglikelihood = logl
        return logl
    loglikelihood.last_parameters = None
    loglikelihood.last_loglikelihood = None

    outdir = "dualanalysis_%s_var%s%s%d" % (
        str(obs['id']).strip(),
        'V' if variability_uncertainty else '',
        'U' if with_uv_model_uncertainty else '',
        -int(np.log10(systematics_width)),
    )
    if args.mass_max != 15:
        outdir += "_maxgal%d" % args.mass_max
    print("Sampling with sampler:", samplername)
    replot = not os.path.exists(outdir + '/analysis_results.txt')
    results = None
    # print("  free parameters:", active_param_names, derived_param_names)
    if samplername == 'laplace':
        from snowline import ReactiveImportanceSampler
        print("Laplace approximation ...")
        sampler = ReactiveImportanceSampler(param_names, loglikelihood, prior_transform)
        # sampler.run(num_global_samples=1000, max_improvement_loops=1)
        sampler.laplace_approximate(num_global_samples=1000)
        sampler.cov = np.eye(len(param_names)) * 0.04
        sampler.invcov = np.linalg.inv(sampler.cov)
        # sampler.init_globally(num_global_samples=1000)
        for sampling_results in sampler.run_iter(
            num_gauss_samples=1000,
            min_ess=1000,
            max_improvement_loops=5,
        ):
            print("Importance sampling ...")
            sampler.print_results()
            np.savetxt(
                outdir + '_laplace.txt.gz', sampling_results['samples'], 
                delimiter=',', comments='', header=','.join(param_names)
            )
            if plot and not os.path.exists(outdir + '_laplace.pdf'):
                results = plot_posteriors(outdir + '_laplace.pdf', prior_samples, param_names, sampling_results['samples'])
    elif samplername == 'mcmc':
        from autoemcee import ReactiveAffineInvariantSampler
        with FastExtinction():
            sampler = ReactiveAffineInvariantSampler(param_names, loglikelihood, prior_transform)
            sampler.run()
            sampler.print_results()
        np.savetxt(
            outdir + '_mcmc.txt.gz', sampler.results['samples'], 
            delimiter=',', comments='', header=','.join(param_names)
        )
        if plot and not os.path.exists(outdir + '_mcmc.pdf'):
            results = plot_posteriors(outdir + '_mcmc.pdf', prior_samples, param_names, sampler.results['samples'])
    elif samplername == 'nested':
        with FastExtinction():
            sampler = ReactiveNestedSampler(
                active_param_names, loglikelihood, prior_transform,
                log_dir=outdir, resume=True, derived_param_names=derived_param_names)
            sampler.run(frac_remain=0.9, max_num_improvement_loops=0, min_num_live_points=400, viz_callback=None)
            sampler.print_results()
        if plot:
            results = plot_results(sampler, prior_samples, obs, obs_fluxes, obs_errors, wobs, cache_filters, replot=replot)
        sampler.pointstore.close()
    elif samplername == 'nested-reactive':
        with FastExtinction():
            sampler = ReactiveNestedSampler(
                active_param_names, loglikelihood, prior_transform,
                log_dir=outdir, resume=True, derived_param_names=derived_param_names)
            sampler.run(frac_remain=0.5, max_num_improvement_loops=5, min_num_live_points=50, min_ess=500, dlogz=10, cluster_num_live_points=0)
            sampler.print_results()
        if plot:
            results = plot_results(sampler, prior_samples, obs, obs_fluxes, obs_errors, wobs, cache_filters, replot=replot)
        sampler.pointstore.close()
    elif samplername == 'nested-slice':
        outdir += "-slice"
        replot = not os.path.exists(outdir + '/analysis_results.txt')
        with FastExtinction():
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
            import ultranest.stepsampler
            print("run without step sampler ...")
            sampler.run(frac_remain=0.5, max_num_improvement_loops=0, min_num_live_points=50, dlogz=10, cluster_num_live_points=0, max_ncalls=10000, viz_callback=None)
            print("run with step sampler ...")
            sampler.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=20,
                generate_direction=ultranest.stepsampler.generate_mixture_random_direction)
            sampler.run(
                frac_remain=0.5, min_num_live_points=50, dlogz=10, cluster_num_live_points=0, 
                min_ess=100, viz_callback=None, max_num_improvement_loops=0,
            )
            sampler.print_results()
        if plot:
            results = plot_results(sampler, prior_samples, obs, obs_fluxes, obs_errors, wobs, cache_filters, replot=replot)
        sampler.pointstore.close()
    elif samplername == 'noop':
        pass
    else:
        raise ValueError("Unknown sampler: '%s'" % samplername)
    if results is not None:
        names, means, stds, los, his = results
        with open(outdir + '/analysis_results.txt', 'w') as fout:
            fout.write("%s" % obs['id'])
            for name, mean, std, lo, hi in zip(names, means, stds, los, his):
                fout.write("\t%g\t%g\t%g\t%g" % (mean, std, lo, hi))
            fout.write('\n')
    try:
        results_string = open(outdir + '/analysis_results.txt', 'r').read()
    except IOError:
        results_string = None
    
    return obs['id'], results, results_string

def main():
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

        # pick observation
        obs_table_here = obs_table[args.offset::args.every]
        indices = np.arange(len(obs_table_here))
        if args.randomize:
            np.random.shuffle(indices)
        fout = None
        # analyse 20 observations in parallel, then reset children
        # this is to avoid excessive memory use (there is a memory leak).
        for i in np.array_split(indices, max(1, len(obs_table_here) // chunk_size)):
            print("processing chunk with indices:", i)
            allresults = joblib.Parallel(n_jobs=args.cores)(
                joblib.delayed(analyse_obs_wrapper)(args.sampler, obs, plot)
                for obs in obs_table_here[i]
            )
            for id, result, results_string in allresults:
                if results_string is None:
                    print("no result to store for", id, ". Delete plots, otherwise results will not be reanalysed.")
                    continue
                print("result", id, result)
                names = param_names + analysed_variables + ['NEV', 'Lbol', 'chi2']
                if fout is None:
                    fout = open(data_file + '_analysis_results.txt', 'w')
                    fout.write('# id')
                    for name in names:
                        fout.write('\t%s_mean\t%s_std\t%s_lo\t%s_hi' % (name, name, name, name))
                    fout.write('\n')
                fout.write(results_string)
                fout.flush()

        #for obs in obs_table:
        #    analyse_obs(samplername, obs, plot=plot)
        print("analying %d observations done." % len(obs_table_here))

if __name__ == '__main__':
    main()

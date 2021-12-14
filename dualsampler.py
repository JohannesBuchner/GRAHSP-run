
import numpy as np
from numpy import log, log10
from math import erf
from pcigale.session.configuration import Configuration
from pcigale.analysis_modules import get_module as get_analysis_module
from pcigale.utils import read_table
from pcigale.analysis_modules import complete_obs_table
from pcigale.warehouse import SedWarehouse
from pcigale import creation_modules
from pcigale.analysis_modules.pdf_analysis import TOLERANCE
from pcigale.data import Database
from ultranest import ReactiveNestedSampler
from ultranest.plot import PredictionBand
import scipy.stats
from scipy.constants import c
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tqdm
import os

gbl_warehouse = SedWarehouse()

config = Configuration("pcigale.ini")

data_file = config.configuration['data_file']
column_list = config.configuration['column_list']
module_list = config.configuration['creation_modules']
parameter_list = config.configuration['creation_modules_params']
cache_depth = module_list.index('extinction')
cache_depth = module_list.index('activate')
cache_max = int(os.environ.get('CACHE_MAX', '10000'))
cache_print = os.environ.get('CACHE_VERBOSE', '0') == '1'
analysis_module = get_analysis_module(config.configuration[
    'analysis_method'])
analysis_module_params = config.configuration['analysis_method_params']

analysed_variables = analysis_module_params["analysed_variables"]
n_variables = len(analysed_variables)
lim_flag = analysis_module_params["lim_flag"].lower() == "true"
mock_flag = analysis_module_params["mock_flag"].lower() == "true"

filters = [name for name in column_list if not name.endswith('_err')]
with Database() as base:
    filters_wl_orig = np.array([base.get_filter(name).effective_wavelength for name in filters])
    """
    filters_range = []
    for name in filters:
        filt = base.get_filter(name)
        filtwave = filt.trans_table[0][filt.trans_table[1] >= filt.trans_table[1].max() / 0.5]
        filters_range.append([filtwave.min(), filtwave.max()])
    """
n_filters = len(filters)


param_names = []
is_log_param = []
print()
print("Parameters")
print("----------")
for module_name, module_parameters in zip(module_list, parameter_list):
    print("  [%s]" % module_name)
    parameter_list_agn_here = {}
    parameter_list_gal_here = {}
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
#param_names.append("systematics")
#rv_systematics = scipy.stats.halfnorm(scale=0.05)

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

def get_model_fluxes(sed):
    if 'sfh.age' in sed.info and sed.info['sfh.age'] > sed.info['universe.age']:
        model_fluxes = -99. * np.ones(len(filters))
        model_variables = -99. * np.ones(len(analysed_variables))
    else:
        model_fluxes = np.array([sed.compute_fnu(filter_) for filter_ in filters])
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
                parameter_list_gal_here[k] = selected_value
                parameter_list_agn_here[k] = selected_value
            elif 'activate' in module_name or 'AGN' in k:
                parameter_list_gal_here[k] = mock_value
                parameter_list_agn_here[k] = selected_value
            else:
                parameter_list_gal_here[k] = selected_value
                parameter_list_agn_here[k] = mock_value
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
    # convert from erg/s to W, the luminosity unit of cigale
    scaled_sed.luminosities[agn_mask] = agn_sed.luminosities[agn_mask] * L_AGN / 1e7
    # copy over AGN meta data
    scaled_sed.info.update({k:v for k, v in agn_sed.info.items() if 'activate' in k})
    # print(scaled_sed.info)

    for module_name, module_parameters in zip(module_list[cache_depth:], parameter_list_gal[cache_depth:]):
        module_instance = creation_modules.get_module(module_name, **module_parameters)
        module_instance.process(scaled_sed)

    #print("IGM:", igm.min(), igm.max())
    assert np.isfinite(scaled_sed.luminosities[:-1].sum(0)).all(), scaled_sed.luminosities[:-1].sum(0)
    # recompute IGM attenuation
    #scaled_sed.luminosities[-1] = igm * scaled_sed.luminosities[:-1].sum(0)
    assert np.isfinite(scaled_sed.luminosities).all(), scaled_sed.luminosities
    
    # update total luminosities
    scaled_sed.luminosity = scaled_sed.luminosities.sum(0)
    assert np.isfinite(scaled_sed.luminosity).all(), scaled_sed.luminosity
    return scaled_sed

#@functools.lru_cache(maxsize=None)

#import joblib
#mem = joblib.Memory('.', verbose=False)

#@mem.cache
#def compute_model_cached(parameters):
#   parameter_list_here = make_parameter_list(parameters)
#   sed, model_fluxes_full, model_variables = compute_model(parameter_list_here)
#   return model_fluxes_full

# Wavelength limits (restframe) when plotting the best SED.
PLOT_L_MIN = 0.1
PLOT_L_MAX = 50

def with_attenuation(keys):
    return keys + ['attenuation.' + key for key in keys]

def plot_results(sampler, obs, obs_fluxes, obs_errors, wobs, cache_filters):
    # only allow the main process to plot
    if not sampler.log:
        return

    results = sampler.results
    plot_dir = sampler.logs['plots']
    prior_samples = sampler.transform(np.random.uniform(size=(10000, len(param_names))))
    assert np.isfinite(prior_samples).all(), (np.where(~np.isfinite(prior_samples).all(axis=0)), np.where(~np.isfinite(prior_samples).all(axis=1)), prior_samples[~np.isfinite(prior_samples)])
    assert np.isfinite(results['samples']).all()
    
    print("plotting posteriors")
    plt.figure(figsize=(12, 12))
    for i, (param_name, samples) in enumerate(zip(param_names, results['samples'].transpose())):
        plt.subplot(4, len(param_names) // 4 + 1, i + 1)
        bins = np.unique(list(set(prior_samples[:,i]).union(set(samples))))
        if len(bins) > 40:
            bins = 20
        plt.hist(samples, histtype='step', density=True, bins=bins)
        xlo, xhi = plt.xlim()
        plt.hist(prior_samples[:,i], histtype='step', 
            density=True, bins=bins, color='gray', ls='-')
        plt.xlim(xlo, xhi)
        plt.yticks([])
        plt.xlabel(param_names[i])
    
    plt.subplots_adjust(wspace=0.1)
    plt.savefig('%s/posteriors.pdf' % plot_dir, bbox_inches='tight')
    plt.close()
    
    bands = {'lum':{}, 'mJy':{}}

    plot_elements = [
        dict(keys=with_attenuation(['stellar.young', 'stellar.old']),
             label="Stellar attenuated", color='orange', marker=None, linestyle='-',),
        #dict(keys=['stellar.young', 'stellar.old'],
        #     label="Stellar unattenuated", color='b', marker=None, nonposy='clip', linestyle='--', linewidth=0.5),
        dict(keys=with_attenuation(['nebular.lines_young', 'nebular.lines_old', 'nebular.continuum_young', 'nebular.continuum_old']),
             label="Nebular emission", color='y', marker=None, linewidth=.5),
        dict(keys=['agn.activate_Disk'],
             label="AGN disk", color=[0.90, 0.90, 0.72], marker=None, linestyle='-', linewidth=1.5),
        dict(keys=['agn.activate_Torus'],
             label="AGN torus", color=[0.90, 0.77, 0.42], marker=None, linestyle='-', linewidth=1.5),
        dict(keys=['agn.activate_EmLines_BL', 'agn.activate_EmLines_NL', 'agn.activate_FeLines', 'agn.activate_EmLines_LINER'],
             label="AGN lines", color=[0.90, 0.50, 0.21], marker=None, linestyle='-', linewidth=0.5),
        #dict(keys=['F_lambda_total'],
        #     label="Model spectrum", color='k', marker=None, nonposy='clip', linestyle='-', linewidth=1.5, alpha=0.7),
    ]
    
    z = obs['redshift']
    chi2_best = -2 * sampler.results['weighted_samples']['logl'].max()
    # chi2_reduced = chi2_best / wobs.sum()
    
    posteriors = []
    
    for parameters in tqdm.tqdm(sampler.results['samples'][:500,:]):
        stellar_mass = 10**parameters[-3]
        L_AGN = 10**parameters[-2]
        redshift = parameters[-1]
        # systematic_flux_error = parameters[-1]
        parameter_list_here = make_parameter_list(parameters)
        parameter_list_here[-1] = dict(redshift=redshift)

        sed = scale_sed_components(module_list, parameter_list_here, stellar_mass, L_AGN)
        sed.cache_filters = cache_filters
        model_fluxes_full, model_variables = get_model_fluxes(sed)
        posteriors.append(model_variables)

        mod_fluxes = model_fluxes_full[wobs]
        
        # print(dir(sed), sed.info.keys())
        wavelength_spec = sed.wavelength_grid
        DL = sed.info['universe.luminosity_distance']

        for sed_type in 'mJy', 'lum':
            wavelength_spec2 = wavelength_spec.copy()
            if sed_type == 'lum':
                sed_multiplier = wavelength_spec2.copy()
                wavelength_spec2 /= 1. + z
            elif sed_type == 'mJy':
                sed_multiplier = (wavelength_spec2 * 1e29 /
                                   (c / (wavelength_spec2 * 1e-9)) /
                                   (4. * np.pi * DL * DL))

            assert (sed_multiplier >= 0).all(), (stellar_mass, DL, wavelength_spec2)
            wavelength_spec2 /= 1000

            for j, plot_element in enumerate(plot_elements):
                keys = plot_element['keys']
                if not all(k in sed.contribution_names for k in keys):
                    continue
                if j not in bands[sed_type]:
                    print("building", sed_type, plot_element['label'])
                    bands[sed_type][j] = PredictionBand(wavelength_spec2)
                pred = sum(sed.get_lumin_contribution(k) * sed_multiplier for k in keys)
                assert np.isfinite(pred).all(), pred
                assert bands[sed_type][j].x.shape == pred.shape, (bands[sed_type][j].x.shape, pred.shape)
                # print(plot_element['label'], pred)
                bands[sed_type][j].add(pred)
            if 'total' not in bands[sed_type]:
                bands[sed_type]['total'] = PredictionBand(wavelength_spec2)
            bands[sed_type]['total'].add(sed.luminosity * sed_multiplier)

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
                print("plotting", sed_type, plot_element['label'], np.shape(bands[sed_type][j].ys))
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
                     markersize=6, markeredgecolor='b', capsize=0.)
        mask_uplim = np.logical_and(np.logical_and(obs_fluxes > 0.,
                                               obs_fluxes_err < 0.),
                                obs_fluxes_err > -9990. * k_corr_SED)

        if not mask_uplim.any() == False:
            ax1.errorbar(filters_wl[mask_uplim], obs_fluxes[mask_uplim],
                         yerr=obs_fluxes_err[mask_uplim]*3, ls='',
                         marker='v', label='Observed upper limits',
                         markerfacecolor='None', markersize=6,
                         markeredgecolor='g', capsize=0.)
        mask_noerr = np.logical_and(obs_fluxes > 0.,
                                    obs_fluxes_err < -9990. * k_corr_SED)
        if not mask_noerr.any() == False:
            ax1.errorbar(filters_wl[mask_noerr], obs_fluxes[mask_noerr],
                         ls='', marker='s', markerfacecolor='None',
                         markersize=6, markeredgecolor='r',
                         label='Observed fluxes, no errors', capsize=0.)
        mask = np.where(obs_fluxes > 0.)
        ax2.errorbar(filters_wl[mask],
                     (obs_fluxes[mask]-mod_fluxes[mask])/obs_fluxes[mask],
                     yerr=obs_fluxes_err[mask]/obs_fluxes[mask]*3,
                     marker='_', label="(Obs-Mod)/Obs", color='k',
                     capsize=0., linestyle=' ')
        ax2.plot([xmin, xmax], [0., 0.], ls='--', color='k')
        ax2.set_xscale('log')
        ax1.set_xscale('log')
        ax2.minorticks_on()

        figure.subplots_adjust(hspace=0.2, wspace=0.)

        ax1.set_xlim(xmin, xmax)
        ax2.set_xlim(xmin, xmax)
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
        if np.isinf(ymin):
            ymin = ymax / 100
        ax1.set_ylim(1e-2*ymin, 1e1*ymax)
        ax1.set_yscale('log')
        ax2.set_ylim(-1.0, 1.0)
        if sed_type == 'lum':
            ax2.set_xlabel("Rest-frame wavelength [$\mu$m]")
            ax1.set_ylabel("Luminosity [W]")
            ax2.set_ylabel("Relative residual luminosity")
        else:
            ax2.set_xlabel("Observed wavelength [$\mu$m]")
            ax1.set_ylabel("Flux [mJy]")
            ax2.set_ylabel("Relative residual flux")
        ax1.legend(fontsize=6, loc='best', fancybox=True, framealpha=0.5)
        ax2.legend(fontsize=6, loc='best', fancybox=True, framealpha=0.5)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels()[1], visible=False)
        figure.suptitle(
            "Best model for %s at z = %.3f. $\chi^2$=%.1f/%d" %
                (obs['id'], obs['redshift'], chi2_best, len(obs_fluxes)))
        figure.savefig("%s/sed_%s.pdf" % (plot_dir, sed_type))
        plt.close(figure)

def make_prior_transform(rv_redshift):
    def prior_transform(cube):
        params = cube.copy()
        i = 0
        for module_parameters in parameter_list:
            for k, v in module_parameters.items():
                if len(v) > 1:
                    params[i] = v[int(len(v) * cube[i])]
                    if is_log_param[i]:
                        params[i] = log10(params[i])
                    i += 1
        
        # stellar mass from 10^5 to 10^15
        params[i] = cube[i] * 10 + 5
        
        # AGN luminosity from 10^38 to 10^50
        params[i+1] = cube[i+1] * 20 + 30

        # redshift. 
        # params[i+1] = rv_redshift.ppf(cube[i+1])
        # Approximate redshift with points on the CDF
        params[i+2] = rv_redshift.ppf((1 + np.round(cube[i+2] * 40)) / 42)
        
        # systematic uncertainty
        # params[i+3] = rv_systematics.ppf(cube[i+3])
        return params
    return prior_transform

def plot_model():
    umid = np.zeros(len(param_names)) + 0.5
    for redshift in [0, 0.5, 1, 3]:
        for logL_AGN in [38, 42, 44, 46]:
            L_AGN = 10**logL_AGN
            rv_redshift = scipy.stats.uniform(redshift, redshift+1e-3)
            prior_transform = make_prior_transform(rv_redshift)
            print("reference values:")
            for p, v in zip(param_names, prior_transform(umid)):
                print("   %-20s: %s" % (p, v))
            cache_filters = {}
            
            for i, p in enumerate(param_names):
                if p in ('redshift', 'log(L_AGN)', 'activate.AGNtype'):
                    continue
                print("varying", p)

                u = umid.copy()
                for AGNtype in 1, 2:
                    last_value = np.nan
                    for v in np.linspace(0, 0.999, 11):
                        u[i] = v
                        parameters = prior_transform(u)
                        parameters[param_names.index('activate.AGNtype')] = AGNtype
                        if parameters[i] == last_value:
                            continue
                        last_value = parameters[i]
                        stellar_mass = 10**parameters[-3]
                            
                        parameter_list_here = make_parameter_list(parameters)
                        assert module_list[-1] == 'redshifting'
                        parameter_list_here[-1] = dict(redshift=redshift)
                        
                        with np.errstate(invalid='ignore'):
                            sed = scale_sed_components(module_list, parameter_list_here, stellar_mass, L_AGN)
                        sed.cache_filters = cache_filters
                        
                        wavelength_spec = sed.wavelength_grid
                        DL = sed.info['universe.luminosity_distance']
                        sed_multiplier = (wavelength_spec * 1e29 /
                                           (c / (wavelength_spec * 1e-9)) /
                                           (4. * np.pi * DL * DL))

                        assert (sed_multiplier >= 0).all(), (stellar_mass, DL, wavelength_spec)
                        wavelength_spec /= 1000
                        mask = np.logical_and(wavelength_spec >= PLOT_L_MIN, wavelength_spec <= PLOT_L_MAX)

                        if AGNtype == 1:
                            plt.plot(wavelength_spec[mask], (sed.luminosity * sed_multiplier)[mask], 
                                color=plt.cm.viridis(v), label=parameters[i], ls='-')
                        else:
                            plt.plot(wavelength_spec[mask], (sed.luminosity * sed_multiplier)[mask], 
                                color=plt.cm.viridis(v), ls='--')
                
                plt.xlabel("Observed wavelength [$\mu$m]")
                plt.ylabel("Flux [mJy]")
                plt.xscale('log')
                plt.yscale('log')
                plt.legend(title=p, fontsize=6, loc='best', fancybox=True, framealpha=0.5)
                plt.xlim(PLOT_L_MIN, PLOT_L_MAX)
                plt.ylim(1e13, None)
                #plt.setp(plt.gca().get_xticklabels(), visible=False)
                #plt.setp(plt.gca().get_yticklabels()[1], visible=False)
                
                filename = 'modelspectrum_z%.1f_L%d_%s.png' % (redshift, logL_AGN, p.replace('(','').replace(')',''))
                plt.savefig(filename, bbox_inches='tight')
                plt.close()
#            break
#       break
        

def generate_fluxes():
    rv_redshift = scipy.stats.uniform(0, 6)
    prior_transform = make_prior_transform(rv_redshift)
    cache_filters = {}

    Ngen = 100000
    fluxdata = np.empty((Ngen, len(param_names) + len(filters)))
    u = np.random.uniform(size=len(param_names))
    for i in tqdm.trange(Ngen):
        u[np.random.randint(len(param_names))] = np.random.uniform()
        parameters = prior_transform(u)
        stellar_mass = 10**parameters[-3]
        L_AGN = 10**parameters[-2]
        redshift = parameters[-1]
        parameter_list_here = make_parameter_list(parameters)
        assert module_list[-1] == 'redshifting'
        parameter_list_here[-1] = dict(redshift=redshift)
        
        sed = scale_sed_components(module_list, parameter_list_here, stellar_mass, L_AGN)
        sed.cache_filters = cache_filters

        model_fluxes_full, model_variables = get_model_fluxes(sed)
        fluxdata[i][:len(param_names)] = parameters
        fluxdata[i][len(param_names):] = model_fluxes_full

    np.savetxt("model_fluxes.txt.gz", fluxdata, header=','.join(param_names + filters), delimiter=',', comments='')

def main(sampler='nested'):
    # Read the observation table and complete it by adding error where
    # none is provided and by adding the systematic deviation.
    obs_table = complete_obs_table(read_table(data_file), column_list,
                                   filters, TOLERANCE, lim_flag)

    # pick observation
    for obs in obs_table:
        np.random.seed(1)
        redshift_mean = obs['redshift']
        if 'redshift_err' in obs.colnames:
            redshift_err = obs['redshift_err']
        else:
            # put a 1% error on 1+z, at least
            redshift_err = 0.1 * redshift_mean
        redshift_samples = np.random.normal(redshift_mean, redshift_err, size=1000)
        redshift_samples = redshift_samples[redshift_samples>0]
        redshift_shape, _, redshift_scale = scipy.stats.weibull_min.fit(
            redshift_samples, floc=0,
        )
        print("redshift:", redshift_shape, redshift_scale)
        rv_redshift = scipy.stats.weibull_min(redshift_shape, scale=redshift_scale)

        prior_transform = make_prior_transform(rv_redshift)
        # select the filters from the list of active filters

        obs_fluxes_full = np.array([obs[name] for name in filters])
        obs_errors_full = np.array([obs[name + "_err"] for name in filters])

        wobs = np.where(obs_fluxes_full > TOLERANCE)
        obs_fluxes = obs_fluxes_full[wobs]
        obs_errors = obs_errors_full[wobs]

        cache_filters = {}
        
        def loglikelihood(parameters):
            stellar_mass = 10**parameters[-3]
            L_AGN = 10**parameters[-2]
            redshift = parameters[-1]
            parameter_list_here = make_parameter_list(parameters)
            assert module_list[-1] == 'redshifting'
            parameter_list_here[-1] = dict(redshift=redshift)
            
            sed = scale_sed_components(module_list, parameter_list_here, stellar_mass, L_AGN)
            sed.cache_filters = cache_filters

            model_fluxes_full, model_variables = get_model_fluxes(sed)

            model_fluxes = model_fluxes_full[wobs]
            
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
            chi2_ = np.sum(np.square(
                (obs_fluxes[mask_data]-model_fluxes[mask_data]) /
                (obs_errors[mask_data])))

            if mask_lim.any():
                chi2_ += -2. * log(
                        np.sqrt(np.pi/2.)*(-obs_errors[mask_lim])*(
                            1.+erf(
                                (obs_fluxes[mask_lim]-model_fluxes[mask_lim]) /
                                (np.sqrt(2)*(-obs_errors[mask_lim]))))).sum()
            #print("chi2:", chi2_, parameters)
            return -0.5 * chi2_

        if sampler == 'laplace':
            from snowline import ReactiveImportanceSampler
            sampler = ReactiveImportanceSampler(param_names, loglikelihood, prior_transform)
            sampler.run(num_global_samples=1000, max_improvement_loops=1)
        if sampler == 'mcmc':
            from autoemcee import ReactiveAffineInvariantSampler
            sampler = ReactiveAffineInvariantSampler(param_names, loglikelihood, prior_transform)
            sampler.run()
            sampler.print_results()
        elif sampler == 'nested':
            sampler = ReactiveNestedSampler(
                param_names, loglikelihood, prior_transform,
                log_dir="dualanalysis_%s_chi2" % obs['id'], resume=True)
            sampler.run(frac_remain=0.5, max_num_improvement_loops=0)
            sampler.print_results()
            try:
                sampler.plot()
            except Exception:
                pass
            plot_results(sampler, obs, obs_fluxes, obs_errors, wobs, cache_filters)
        
        

if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'generate-from-prior':
        generate_fluxes()
    elif sys.argv[1] == 'plot-model':
        plot_model()
    else:
        main(sys.argv[1])

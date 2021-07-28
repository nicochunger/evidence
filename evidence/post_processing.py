import os
import matplotlib
# If running in server the 'Agg' display enviornment has to be used
home = os.getenv('HOME')
if 'astro' in home:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from pathlib import Path
import importlib
import sys
import numpy as np
import pandas as pd
import scipy.stats as st
import pickle
from evidence.rvmodel import RVModel
from corner import corner


def postprocess(path, order=True, plotcorner=False):
    """ Script that does a post processing of the nested sampling output. It analyses
    posterior samples and produces phase fold plots for the planets.
    It works with the output of both PolyChord and UltraNests. """

    # -------------------- SETUP ------------------------------
    # Write all prints to a file to save for later
    save_path = os.path.join(path, 'results.txt')
    f = open(save_path, 'w')

    # Load pickle file with polychord output
    print('Loading pickle file with Nested Sampling output...')
    run_label = str(path).split('/')[-1]
    output = pickle.load(
        open(os.path.join(path, run_label+'.pkl'), 'rb'), encoding='latin1')
    print(f'Run: {output.file_root}', file=f)

    rundict_keys = list(output.rundict.keys())

    # Print run parameters
    print('Saving run parameters and results...')
    print('\nRun parameters', file=f)
    print(f'Target: {output.rundict["target"]}', file=f)
    print(f'Run ID: {output.rundict["runid"]}', file=f)
    if 'comment' in rundict_keys:
        print(f'Comment: {output.rundict["comment"]}', file=f)
    print(f'Live points: {output.nlive} ({output.nlive//output.ndim} * ndim)', file=f)
    print(f'Repeats in slice sampling: {output.nrepeats} ({output.nrepeats//output.ndim} * ndim)', file=f)
    print(f'Cores used: {output.ncores}', file=f)
    print(f'Run time: {output.runtime}', file=f)
    print(f'Likelihood calculations: {output.nlike}', file=f)
    print(f'Sampler: {output.sampler}', file=f)

    nplanets = None
    if 'nplanets' in rundict_keys:
        nplanets = output.rundict["nplanets"]
        if nplanets != None:
            print(f'\nNumber of planets: {nplanets}', file=f)
    print(f'Number of free parameters: {output.ndim}', file=f)

    # Run results
    print('\nResults:', file=f)
    print(f'Evidence (logZ) = {output.logZ:.3f} +/- {output.logZerr:.3f}', file=f)

    # Change direcory of posterior
    if output.sampler == 'PolyChord':
        output.base_dir = os.path.join(path, 'polychains')
    elif output.sampler == 'UltraNest':
        output.base_dir = os.path.join(path, 'ultraresults')

    # Samples of posterior
    parnames = output.parnames
    # posterior = output.posterior
    # weights = posterior.weights
    # weights /= sum(weights) # Normalize weights
    # Samples construction. Using equally weighted posterior samples
    samples = output.samples
    if output.sampler == 'PolyChord':
        loglikes = samples['loglike']
        del output.samples['loglike']
        del output.samples['weight']
        old_cols = output.samples.columns.values.tolist()
        output.samples.rename(columns=dict(zip(old_cols, parnames)), inplace=True)
    elif output.sampler == 'UltraNest':
        wp_file = os.path.join(output.base_dir, 'run1/chains/weighted_post.txt')
        weighted_post = pd.read_csv(wp_file, sep=' ')
        loglikes = weighted_post['logl']
        weights = weighted_post['weight']

    print(f'\nNr. of samples in posterior: {len(samples)}', file=f)

    # ------------- ORDER POSTERIOR SAMPLES ---------------------
    if order:
        planets = [] # Get all idxs of the parameters for each planet
        planet_idxs = [] # Positions of the periods of all planets
        for n in range(1, nplanets+1):
            planets.append([])
            for i, par in enumerate(parnames):
                if f'planet{n}' in par:
                    planets[n-1].append(i)
                    if 'period' in par:
                        planet_idxs.append(i)

        # Swapping the planet samples so that the periods are always ordered
        # TODO If at some point this is too slow a vectorized version has to be implemented
        for idx, sample in samples.iterrows():
            periods_tmp = np.zeros(nplanets)
            sample_arr = sample.values.copy()
            periods_tmp = sample_arr[planet_idxs]
            # print(periods_tmp)
            idxs = np.arange(len(sample), dtype=int)
            # for n in range(1, nplanets+1):
            #     periods_tmp[n-1] = sample[f'planet{n}_period']
            if not np.all(periods_tmp[:-1] <= periods_tmp[1:]):
                # Sort periods
                sorted_periods_args = np.argsort(periods_tmp)
                # Construct target index list
                for i, par in enumerate(parnames):
                    if 'planet' not in par:
                        idxs[i] = i
                    else:
                        planet = int(par[6])
                        new_pos = list(sorted_periods_args).index(planet-1)
                        internal_pos = planets[planet-1].index(i)
                        target = planets[new_pos][internal_pos]
                        idxs[i] = target
                # print(idxs)
            samples.iloc[idx] = sample_arr[idxs]
    # -----------------------------------------------------------

    medians = np.median(samples, axis=0)
    stds = np.std(samples, axis=0)

    # Initialize DataFrame
    params = pd.DataFrame(index=parnames)
    params['Median'] = medians
    params['Std'] = stds
    try:
        # Add prior for each parameter
        params['Prior'] = pd.Series(output.priors)
    except:
        pass

    print(f'Median, error and prior for each parameter', file=f)
    pd.set_option("display.max_columns", 4)
    print(params, file=f)
    print(params)

    print('\n============== Planet parameters ==================', file=f)

    # Print planet parameters
    if (nplanets != None) and (nplanets != 0):
        for i in range(nplanets):
            try:
                from uncertainties import ufloat, umath
                # Planet properties calculations
                K = ufloat(params['Median'][f'planet{i+1}_k1'], params['Std'][f'planet{i+1}_k1'])
                period = ufloat(
                    params['Median'][f'planet{i+1}_period'], params['Std'][f'planet{i+1}_period'])

                # Eccentricity extraction
                if f'planet{i+1}_ecc' in parnames:
                    ecc = ufloat(
                        params['Median'][f'planet{i+1}_ecc'], params['Std'][f'planet{i+1}_ecc'])
                else:
                    secos = ufloat(params['Median'][f'planet{i+1}_secos'], params['Std'][f'planet{i+1}_secos'])
                    sesin = ufloat(params['Median'][f'planet{i+1}_sesin'], params['Std'][f'planet{i+1}_sesin'])
                    ecc = secos**2 + sesin**2

                # Print planet parameters to results file
                mstar = ufloat(
                    output.starparams['star_mass'][0], output.starparams['star_mass'][1])
                print(f"\nPlanet {i+1}", file=f)
                print(f"Period = {period}", file=f)
                print(f'Semi amplitude = {K}', file=f)
                print(f'Eccentricity = {ecc}', file=f)
                print(
                    f"m*sin(i) = {min_mass(K, period, ecc, mstar)} Mearth", file=f)
                print(f"a = {semi_mayor_axis(period, mstar)} AU", file=f)
            except (ImportError, KeyError, AttributeError):
                print("No planet parameters could be extracted because of missing key")

    # --------------------------------------------------------


    # ------------ MAXIMUM LIKELIHOOD ------------------------
    print('\n============== Maximum Likelihood Point ==================', file=f)

    print(f'\nMax log Likelihood = {loglikes.iloc[-1]}\n', file=f)

    print(samples.iloc[-1], file=f)

    # Done with printing, close file
    f.close()
    # --------------------------------------------------------

    samples.to_csv('samples.csv', index=False)

    # --------------- POSTERIORS -----------------------------


    # Plotting posteriors for each parameter category
    print('\nPlotting posterior distributions...')
    # Identify parameter categories
    categories = {}
    for par in parnames:
        category = par.split('_')[0]
        parameter = par.split('_')[1]
        if category not in categories.keys():
            categories.update({category: [parameter]})
        else:
            categories[category].append(parameter)

    # Create subplots for each parameter category
    for cat in categories.keys():
        print(f"\tPlotting posterior of category '{cat}'")
        pars = categories[cat]
        fig, axs = plt.subplots(1, len(pars), figsize=(6*len(pars), 5), constrained_layout=True)
        fig.suptitle(f'{run_label}\n{cat}', fontsize=14)
        axs = np.atleast_1d(axs)  # To support 1 planet models

        for i, par in enumerate(pars):
            if par == 'secos':
                par = 'ecc'
                secos = samples[f'{cat}_secos']**2
                sesin = samples[f'{cat}_sesin']**2
                par_post = secos**2 + sesin**2
            elif par == 'sesin':
                par = 'omega'
                secos = samples[f'{cat}_secos']**2
                sesin = samples[f'{cat}_sesin']**2
                par_post = np.arctan2(sesin, secos)
            else:
                par_post = samples[f'{cat}_{par}']

            median = np.median(par_post)

            print(f"\t\tPlotting parameter '{par}'")

            if ('planet' in cat) and ('period' in par):
                n, bin_edges = np.histogram(par_post, bins='auto')
                logbins = np.logspace(np.log10(bin_edges[0]),np.log10(bin_edges[-1]),len(bin_edges))

                axs[i].hist(par_post, label='Posterior', bins=logbins,
                                        histtype='step', density=True)

                axs[i].set_xscale('log')

                # To avoid scientific notation in the tick labels
                axs[i].xaxis.set_major_formatter(ScalarFormatter())
                axs[i].xaxis.set_minor_formatter(ScalarFormatter())
            else:
                n, bin_edges, _ = axs[i].hist(par_post, label='Posterior', bins='auto',
                                                histtype='step', density=True)

            # Setting xlim so the histogram looks good and constrained
            # Cutting to a certain cutoff level
            cutoff = 0.0001
            left_edge = 0
            for y, x in zip(n, bin_edges[:-1]):  
                if y > cutoff*max(n): 
                    left_edge = x 
                    break
            right_edge = 0
            for y, x in zip(n[-1::-1], bin_edges[-1:0:-1]):  
                if y > cutoff*max(n): 
                    right_edge = x 
                    break

            axs[i].set_xlim([left_edge, right_edge])

            axs[i].set_title(f"{par.capitalize()}\nMedian = {median:5.3f}")
            axs[i].set_xlabel(par)
            if i == 0:
                axs[i].set_ylabel('PDF')

        # Save figure
        # fig.tight_layout()
        if 'planet' in cat:
            filename = f"{cat}_{params['Median'][f'{cat}_period']:.2f}_posteriors.png"
        else:
            filename = f'{cat}_posteriors.png'

        
        fig.savefig(os.path.join(path, filename), dpi=300)

    # In addition to the category posterior plots, one more plot with only the
    # periods of the planets

    try:
        if (nplanets != None) and (nplanets != 0):
            print("\tPlotting posterior for planet periods")
            # Plot posterior for the period of each planet
            fig, axs = plt.subplots(1, nplanets, figsize=(6*nplanets, 5))
            axs = np.atleast_1d(axs)  # To support 1 planet models
            plt.ticklabel_format(style='plain', axis='x')

            # MAIN LOOP
            for i in range(nplanets):
                period_post = samples[f'planet{i+1}_period']
                # weighted_mean = np.average(period_post, weights=weights)
                median = params['Median'][f'planet{i+1}_period']

                # Histogram
                _, bins = np.histogram(period_post, bins='auto')
                logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

                axs[i].hist(period_post, label='Posterior', bins=logbins,
                                        histtype='step', density=True)


                axs[i].set_title(f"Median = {median:5.3f} days")
                # axs[i].set_title(f"Median = {weighted_mean:5.3f} days")
                axs[i].vlines(median, 0, 1, transform=axs[i].get_xaxis_transform(), colors="red", linewidth=0.5)
                axs[i].set_xlabel('Period [d]')
                axs[i].set_xscale('log')

                # To avoid scientific notation in the tick labels
                axs[i].xaxis.set_major_formatter(ScalarFormatter())
                axs[i].xaxis.set_minor_formatter(ScalarFormatter())

                if i == 0:
                    axs[i].set_ylabel('PDF')

            # Save figure
            fig.tight_layout()
            
            fig.savefig(os.path.join(path, 'period_posteriors.png'), dpi=300)
    except KeyError:
        print("Couldn't plot posterior for planets because of missing keys.")
    # -------------------------------------------------------


    # -------------------- CORNER PLOT -------------------------
    if plotcorner:
        print('Plotting corner plot...')
        corner_plot = corner(samples, labels=parnames, bins=40, show_titles=True,
                            quantiles=[0.02275, 0.1585, 0.5, 0.8415, 0.97725])
        corner_plot.savefig(os.path.join(path, 'corner_plot.png'), dpi=300)
    # ----------------------------------------------------------


    # ------------------- PHASE FOLDING ------------------------
    # TODO This entire section should be rewritten to something cleaner
    # Create phase fold plots for all planets in the model.
    # Load model
    model = load_model(output, parnames)
    # Check is model is an instance of RVModel
    if isinstance(model, RVModel) and (nplanets != None) and (nplanets != 0):

        print('Plotting phase folds for the planets...')
        # Construct pardict with the median of each parameter
        pardict = {}
        for key in params.index:
            pardict[key] = params['Median'][key]
        pardict.update(model.fixedpardict)

        # Initialize figure
        fig2, ax2 = plt.subplots(
            1, nplanets, figsize=(6*nplanets, 5), sharey=True)
        ax2 = np.atleast_1d(ax2)

        # Color map to be used for the different instruments
        colors = plt.get_cmap('Dark2').colors

        # Get plot order of planets, from lowest to highest period
        periods = np.zeros(nplanets)
        for i in range(nplanets):
            periods[i] = params['Median'][f'planet{i+1}_period']
        periods_pos = np.argsort(periods)

        # Loop through the planets
        for n in range(1, nplanets+1):
            period = pardict[f'planet{n}_period']
            idx = periods_pos[n-1]  # Get index in subplot for each planet
            t_ref = 0  # Initialization of reference time for the phase fold
            full_prediction = np.zeros_like(model.time)
            full_phases = np.zeros_like(model.time)

            # Go through each instrument and plot
            # This is done so each instrument has a different color in the plot
            for i, instrument in enumerate(model.insts):

                inst_idxs = np.where(model.data['inst_id'] == i)
                # Create arrays with time and data for this instrument
                t = model.time[inst_idxs]
                y = model.vrad[inst_idxs]

                # Make RV prediction exluding the specified planet
                prediction = model.kep_rv(pardict, t, exclude_planet=n)
                # if model.drift_in_model:
                #     drift_prediction = model.drift(pardict, t)
                # else:
                #     drift_prediction = np.zeros_like(t)
                # corrected_data = y - prediction - drift_prediction - \
                #                  pardict[f'{instrument}_offset']

                corrected_data = y - pardict[f'{instrument}_offset']
                corrected_data -= prediction
                if model.drift_in_model:
                    corrected_data -= model.drift(pardict, t)
                if model.linpar_in_model:
                    for linpar in model.linpar_dict:
                        corrected_data -= pardict[f'linpar_{linpar}'] * model.linpar_dict[linpar][inst_idxs]
                planet_prediction = model.modelk(pardict, t, planet=n)
                full_prediction[inst_idxs] += planet_prediction

                # Calculate error with jitter
                yerr = model.svrad[inst_idxs]
                if model.jitter_in_model:
                    errs = np.sqrt(yerr**2 + pardict[f'{instrument}_jitter']**2)
                else:
                    errs = yerr

                # Calculate reference point with maximum
                if t_ref == 0:
                    t_ref = t[np.argmax(planet_prediction)]
                # Fold time array
                phases = (((t - t_ref) / period) % 1. - 0.5) * period
                # Append to full list of phases
                full_phases[inst_idxs] = phases

                # Plot data and prediction in corresponding subplot
                ax2[idx].errorbar(phases, corrected_data, yerr=errs,
                                    fmt='.', label=f'{instrument}',
                                    color=colors[i], zorder=i,
                                    elinewidth=1, barsabove=True)

            # Sort full arrays to plot prediction on top
            sortIndi = np.argsort(full_phases)
            full_phases = full_phases[sortIndi]
            full_prediction = full_prediction[sortIndi]

            ax2[idx].plot(full_phases, full_prediction,
                            color='k', zorder=i+1)
            ax2[idx].set_title(f"Planet {n}; Period: {period:.4f} days")
            ax2[idx].set_xlabel("Orbital phase [days]")
            if idx == 0:
                ax2[idx].set_ylabel("Radial velocity [m/s]")

        ax2[nplanets-1].legend(loc='best')
        fig2.tight_layout()
        fig2.savefig(os.path.join(path, 'phase_folds.png'), dpi=300)
    else:
        print("No planets to make phase folded plots.")
    # -------------------------------------------

    # plt.show()
    print("Done!")

    return


def load_model(output, parnames):
    """ Load model file with data and everything. """

    # Import model file
    model_path = Path(output.base_dir).parent
    sys.path.append(model_path)
    mod = importlib.import_module(output.model_name)

    # Initialize the model with the data and datadict
    model = mod.Model(output.fixedpardict, output.datadict, parnames)

    return model


def int_hist(hist, bin_edges):
    res = 0
    lim95 = 0
    counter1 = 1
    lim99 = 0
    counter2 = 1
    for i in range(len(hist)):
        width = bin_edges[i+1] - bin_edges[i]
        res += hist[i] * width
        if res > 0.95 and counter1:
            lim95 = bin_edges[i]
            counter1 = 0
        if res > 0.99 and counter2:
            lim99 = bin_edges[i]
            counter2 = 0

    return res, lim95, lim99


# MINIM MASS
def min_mass(K, period, ecc, mstar):
    """ 
    Calculates de minimum mass using known parameters for the planet and the star
    K [m/s]: radial velocity amplitud 
    period [days]: orbital period of planet 
    ecc: eccentricity of planet
    mstar [Msun]: Mass of star in Sun masses
    """

    from uncertainties import ufloat, umath
    # Unit conversions
    period *= 86400  # days to seconds
    mstar *= 1.9891e30  # Sun masses to kg

    msini = K*(mstar)**(2/3) * \
        (period/(2*np.pi*6.67e-11))**(1/3) * umath.sqrt(1-ecc**2)

    # Convert to Earth masses
    msini /= 5.972e24
    return msini


def semi_mayor_axis(P, mstar):
    """ 
    Calculates the semimayor axis for a planet.
    P [days]: orbital period of planet in days
    mstar [Msun]: Mass of star in Solar Masses
    """
    # Unit conversion
    P *= 86400  # days to seconds
    mstar *= 1.9891e30  # Msun to kg
    G = 6.67e-11

    sma = ((P**2 * G * mstar) / (4*np.pi**2))**(1/3)

    # In AU
    sma /= 1.496e+11

    return sma


if __name__ == "__main__":
    import argparse
    # Parsing of arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ordering", help='Wether to order the posterior samples according to the planet periods.', action="store_false")
    parser.add_argument("--plotcorner", help="Wether to make a corner plot of all parameters", action='store_true')
    args = parser.parse_args()
    postprocess(Path(__file__).parent.absolute(), order=args.no_ordering, plotcorner=args.plotcorner)

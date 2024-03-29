#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
How to use this script: 

This file will automatically be placed in the parent folder of the Nested Sampling 
runs of a single target and runid. Open a terminal in that folder and
run "python fip_criterion.py". The script will automatically detect which planet
models finished running and how many iterations were done. It then computes the
FIP periodogram (Hara et al. 2021), plots it, and prints some results to a file.

Please refer to the help for more information on additional options of the script.
"""

# Python imports
import os
import sys
import pickle
from pathlib import Path
import warnings
# Dependencies
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from copy import deepcopy
import datetime
import argparse

# Parsing of arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-n", help="Maximum number of planets to use FIP periodogram. \
                Default is 0 and this will get the maximum planets available.", type=int, default=0)
parser.add_argument(
    "-hp", help="Number of peaks to highlight. Default is equal to max number of planets.", type=int, default=0)
parser.add_argument(
    "--recalculate-fip", help="Wether to recalculate the FIP periodogram", action='store_true')
parser.add_argument(
    "--with-alias", help="Whether to include aliases in the calculation of the FIP", action='store_true')
parser.add_argument(
    "--show-fig", help="Whether to show the FIP periodogram figure", action='store_true')
args = parser.parse_args()

dirname = Path(__file__).parent.absolute()
split_dirname = str(dirname).split('/')

runid = split_dirname[-1]
target = split_dirname[-2]

# Loading of all posteriors
# Get name of folders
runs = [name for name in os.listdir(dirname) if os.path.isdir(name)]

# Sort the runs
runs.sort()

# Maximum number of planets to use
maxplanets = args.n
if maxplanets == 0:
    maxplanets = None
elif maxplanets < 0:
    parser.error("The maximum number of planets must be a positive integer.")


def get_period_columns(output):
    """ Gets the indices of the columns in the posterior samples where all the
    planet periods are. """

    columns = output.parnames
    idxs = [] # Indices for the columns where the planet periods are located
    for i, col in enumerate(columns):
        if ('period' in col) and ('planet' in col):
            idxs.append(i)

    return np.array(idxs)

def get_finished_runs(runs, max_npla=None):
    """ 
    Looks at all runs from the selection runlabel and keeps the ones that 
    finished and figures out the minimum number of repeats common to all planet
    number models.
    Returns:
    runs : The filtered list of runs which have finished
    nmod : Total number of planet models (is one more than maximum planets)
    min_iters : Minimum number of iterations to take for each model
    """

    # Remove runs which haven't finished yet
    runs = np.array(runs, dtype=object)
    finished = np.zeros_like(runs, dtype=bool)
    for idx, run in enumerate(runs):
        # Check if target name in folder name, if not its not a run folder
        if target not in run:
            continue
        # Check that planet number does not exceed maximum planet number
        if max_npla != None:
            if int(run.split('_')[2][1]) > max_npla:
                continue
        # Check that the run converged and finished
        if os.path.isfile(os.path.join(run, 'results.txt')):
            # Check that there actually samples in the run
            with open(os.path.join(run, 'results.txt'), 'r') as f:
                lines = f.readlines()
                line_check = "Nr. of samples in posterior: "
                for line in lines:
                    if line_check in line:
                        nr_samples = int(line.strip(line_check))
                        if nr_samples > 0:
                            finished[idx] = True
                            break

    # Only keep finished runs
    runs = runs[finished]

    # Figure out number of planet models (will be one more than maximum nr planets)
    nmod = 0
    iters = []
    for run in runs:
        nplas = int(run.split('_')[2][1])
        if nplas+1 > nmod:
            nmod = nplas+1
            iters.append(1)
        else:
            iters[nplas] += 1

    min_iters = min(iters)

    return runs, nmod, min_iters

runs, nmod, min_iters = get_finished_runs(runs, max_npla=maxplanets)

# Set the correct number of maxplanets if it was none or too large
if maxplanets is None:
    maxplanets = nmod -1
elif maxplanets > nmod - 1:
    warnings.warn(f"Maximum number of planets specified ({maxplanets}) is larger" +
                  f" than the maximum available planets ({nmod-1}). Continuing with" +
                  f" maxplanets = {nmod-1}.", Warning)
    maxplanets = nmod - 1
print(f"Using Max planets = {maxplanets}")
# File to save relevant data
save_path = os.path.join(dirname, f'results_{target}_{runid}_maxpla{maxplanets}.txt')
f = open(save_path, 'w')

print(f'Target: {target}', file=f)
print(f'Runid+comment: {runid}', file=f)
print(f'Maximum number of planets = {nmod-1}', file=f)
print(f'Minimum number of common iterations = {min_iters}', file=f)


template_post = [{'k': n} for n in range(nmod)]

# Posterior list. Each element is one "realization" of the criterion
# Each element has to have the posterior samples of each planet model
posteriors = [deepcopy(template_post) for i in range(min_iters)]

for run in runs:
    # Get number of planets
    nplanets = int(run.split("_")[2][1])

    inserted = False
    idx = 0
    while idx < min_iters and not inserted:
        if 'logZ' in list(posteriors[idx][nplanets].keys()):
            idx += 1
        else:
            # Load output file
            output = pickle.load(
                            open(os.path.join(dirname, run, run+'.pkl'), 'rb'),
                            encoding='latin1')
            
            # Change the directory for each sampler
            if output.sampler == 'PolyChord':
                output.base_dir = os.path.join(dirname, run, 'polychains')
            elif output.sampler == 'UltraNest':
                output.base_dir = os.path.join(dirname, run, 'ultraresults')

            # If model is with 0 planet there are no period samples, so just save
            # evidence and runtime
            if nplanets == 0:
                posteriors[idx][nplanets].update({'logZ': output.logZ})
                posteriors[idx][nplanets].update({'runtime': output.runtime})
            else:
                # Get columns with planet periods
                periods_idxs = get_period_columns(output)
                # Extract posterior smaples with their weights
                if output.sampler == 'PolyChord':
                    run_samples = output.posterior.samples[:, periods_idxs]
                    weights = output.posterior.weights
                elif output.sampler == 'UltraNest':
                    wp_file = os.path.join(output.base_dir, 'run1/chains/weighted_post.txt')
                    weighted_post = pd.read_csv(wp_file, sep=' ')
                    run_samples = weighted_post.iloc[:, periods_idxs+2].values
                    weights = weighted_post['weight'].values

                posteriors[idx][nplanets].update({'samples': run_samples})
                posteriors[idx][nplanets].update({'weights': weights})
                posteriors[idx][nplanets].update({'logZ': output.logZ})
                posteriors[idx][nplanets].update({'runtime': output.runtime})
            inserted = True
    print(f"Done with run {run}")


# Get total observation time for this target
instruments = list(output.datadict.keys())
min_time = 1e9
max_time = 0
for inst in instruments:
    try:
        mint = min(output.datadict[inst]['data']['rjd'].values)
        maxt = max(output.datadict[inst]['data']['rjd'].values)
    except:
        mint = min(output.datadict[inst]['data']['jdb'].values)
        maxt = max(output.datadict[inst]['data']['jdb'].values)
    if mint < min_time:
        min_time = mint
    if maxt > max_time:
        max_time = maxt

Tobs = max_time - min_time

# Get range of periods explored
period_prior_str = output.rundict['prior_names'][f'planet{maxplanets}_period']
period_range = period_prior_str.split(': ')[1][1:-1].split(', ')
Pmin = float(period_range[0])
Pmax = float(period_range[1])
nfreq = 50000
coef_window_fap = 1.

# Construct frequencies array on which on calculate the FIP
nu = np.linspace(2*np.pi/Pmax, 2*np.pi/Pmin, nfreq)
nu_window = coef_window_fap*2*np.pi/Tobs
nua = nu-nu_window/2
nub = nu+nu_window/2

# Probability of there being k planets given the data
logZs = np.zeros(nmod)
stdlogZs = np.zeros(nmod)
runtime = np.zeros(nmod, dtype=datetime.timedelta)
stdruntime = np.zeros(nmod, dtype=datetime.timedelta)
# This will be done with the median of the evidences of all runs
# First extract all evidences for each planet model
for nplanets in range(nmod):
    # Collect all the evidences for a specific planet model
    nlogzs = []
    nruntime = []
    for r in range(min_iters):
        try:
            nlogzs.append(posteriors[r][nplanets]['logZ'])
        except KeyError as e:
            print(f"It failed at run{r} k{nplanets}")
        nruntime.append(posteriors[r][nplanets]['runtime'])

    # Save the exp of the mean of the logs to have the actual evidence Z
    logZs[nplanets] = np.median(nlogzs)
    stdlogZs[nplanets] = np.std(nlogzs)

    runtime[nplanets] = sum(nruntime, datetime.timedelta(0))/min_iters
    squaresum = [datetime.timedelta(seconds=(nruntime[i] - runtime[nplanets]).total_seconds()**2) for i in range(min_iters)]
    stdruntime[nplanets] = datetime.timedelta(seconds=np.sqrt((sum(squaresum, datetime.timedelta(0))/min_iters).total_seconds()))

# Probabilities
logpky = logZs - logsumexp(logZs)
pky = np.exp(logpky)  # Normalization

# --------------------------------------

# Create dataframe for evidence results
evidence_results = pd.DataFrame(columns=['log(Z)', 'std', 'delta_log(Z)', 'p(k|y)', 'runtime', 'rstd'])
evidence_results = evidence_results.rename_axis('Planets')

# Calculate delta in logZ
deltaZ = np.zeros(nmod)
deltaZ[1:] = logZs[1:] - logZs[:-1]

evidence_results['log(Z)'] = logZs
evidence_results['std'] = stdlogZs
evidence_results['delta_log(Z)'] = deltaZ
evidence_results['p(k|y)'] = pky
evidence_results['runtime'] = runtime
evidence_results['rstd'] = stdruntime

# Format numbers
evidence_results['log(Z)'] = evidence_results['log(Z)'].map('{:.2f}'.format)
evidence_results['std'] = evidence_results['std'].map('{:.2f}'.format)
evidence_results['p(k|y)'] = evidence_results['p(k|y)'].map('{:.2e}'.format)


print(evidence_results)
print('', file=f)
print(evidence_results, file = f)

# Save file with evidence results
evidence_results.to_csv(f'evidences_{target}_{runid}_maxpla{maxplanets}.csv', index=False)

# Create directory for the fipnus
os.makedirs(f'fipnus', exist_ok=True)
fapnu_dir = f'fipnus/fipnu_{target}_{runid}_maxpla{maxplanets}.txt'
np.savetxt("fipnus/nu.txt", nu)

if (not os.path.isfile(fapnu_dir)) or args.recalculate_fip:
    # Matrix for fapnus. Rows are each run, columns the individial frequencies
    fapnu = np.ones([min_iters, nfreq])
    # Loop over runs
    nsamps = []
    for run in range(min_iters):
        # Loop over models (except the 0 planet one)
        for kmod in range(1,nmod):
            samples = posteriors[run][kmod]['samples']
            weights = posteriors[run][kmod]['weights']
            weights /= np.sum(weights)  # Normalize weights
            nsamples = samples.shape[0]
            nsamps.append(nsamples)

            for i, x in enumerate(samples):
                if not args.with_alias:
                    x_freqs = 2*np.pi/x
                else:
                    x_freqs[0,:] = 2*np.pi/x
                    x_freqs[1,:] = np.abs(2*np.pi/x + 2*np.pi/0.99727)
                    x_freqs[2,:] = np.abs(2*np.pi/x - 2*np.pi/0.99727)
                    x_freqs[3,:] = np.abs(2*np.pi/x + 2*np.pi/30)
                    x_freqs[4,:] = np.abs(2*np.pi/x - 2*np.pi/30)
                    #x_freqs[3,:] = np.abs(2*np.pi/x + 2*np.pi/365.25)
                    #x_freqs[4,:] = np.abs(2*np.pi/x - 2*np.pi/365.25)
                    # Remove frequencies outside the considered range
                    x_freqs = x_freqs.flatten()
                    x_freqs = x_freqs[x_freqs <= 2*np.pi/Pmin]
                    x_freqs = x_freqs[x_freqs >= 2*np.pi/Pmax]
                # x is the vector of planets mean motions
                beg = np.searchsorted(nub, x_freqs, 'right')
                end = np.searchsorted(nua, x_freqs, 'left')
                listind = []
                for bi, ei in zip(beg, end):
                    listind += range(bi, ei)
                fapnu[run, listind] -= pky[kmod]*weights[i]

    np.savetxt(fapnu_dir, fapnu)
else:
    fapnu = np.loadtxt(fapnu_dir)
    fapnu = np.atleast_2d(fapnu)

freq_periods = 2*np.pi/nu

# ----------------- Convergence TEST ----------------

# Test if runs converged.
# I calculated the FIP for each independent run and now I will check if at each
# frequency the difference between the maximum and the minimum is less than 1 in
# log10

fipnu_cut = np.maximum(fapnu, 1e-15)

log10fips = np.log10(fipnu_cut)

minimums = np.min(log10fips, axis=0)
maximums = np.max(log10fips, axis=0)

diffs = maximums - minimums

check = np.where(diffs > 1)


if len(diffs[check]) > 0:
    print("Convergence not achieved!!!")
    print(f'Periods where it failed: {2*np.pi/nu[check]}')
    convergence_fails = pd.DataFrame(columns=['Periods', 'max-min in FIP'])
    convergence_fails['Periods'] = 2*np.pi/nu[check]
    convergence_fails['max-min in FIP'] = diffs[check]
    print(convergence_fails)
    print('', file=f)
    # print(convergence_fails, file=f)
else:
    print("Convergence achieved!")

# ---------------------------------------------------

logfapnu_means = np.median(log10fips, axis=0)
logfapnu_stds = np.std(log10fips, axis=0)

# Set number of peaks to highlight
if args.hp == 0:
    highlighted_peaks = nmod - 1
    save_tag = None
else:
    highlighted_peaks = args.hp
    save_tag = f'_peaks{args.hp}'

fipnu_mean = np.mean(np.maximum(fapnu, 1e-15), axis=0)

# Nathans FIP plot
from evidence import fip_plots
fipplot = fip_plots.FIP_Plots(np.flip(nu), np.flip(fipnu_cut[0,:]))
fipplot.starname = target
print(f"I'm inputting {maxplanets} as maxpla in the plot_clean function!!!")
_, _, peaks_period, peaks_fip = fipplot.plot_clean(highlighted_peaks, maxplanets, save=True, save_tag=save_tag)

# Save peaks and FIP values of the peaks
fip_peaks = pd.DataFrame(columns=['period', '-log10(fip)', 'fip'])
fip_peaks['period'] = peaks_period
fip_peaks['-log10(fip)'] = peaks_fip
fip_peaks['fip'] = 10**(-peaks_fip)
# Print to results file
print('\nFIP peaks:', file=f)
print(fip_peaks, file=f)


# Save a file with the FIP peaks
fip_peaks_all = pd.DataFrame(columns=['period', '-log10(fip)', 'fip'])
fip_peaks_all['period'] = 2*np.pi / fipplot.omega_peaks
fip_peaks_all['-log10(fip)'] = fipplot.peakvalues
fip_peaks_all['fip'] = 10**(-1*fipplot.peakvalues)
# Sort by FIP values
fip_peaks_all.sort_values('-log10(fip)', ascending=False, inplace=True)
# Save
fip_peaks_all.to_csv(f'fippeaks_{target}_{runid}_maxpla{maxplanets}.csv', index=False)


# Show plot if indicated to do so
if args.show_fig:
    plt.show()
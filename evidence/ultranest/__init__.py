import os
import sys
import importlib
import datetime
import time
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

from evidence.post_processing import postprocess

# PolyChord imports
try:
    from ultranest import ReactiveNestedSampler
    import ultranest.stepsampler
except ImportError:
    raise ImportError("Install UltraNest to use this module.")

# MPI imports
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    print("Install mpi4py to use with OpenMPI. Continuing with one thread.")
    rank = 0
    size = 1


def run(model, rundict, priordict, ultrasettings=None):
    """ 
    Runs PolyChord on the chosen data and model. When PolyChord is finished
    running it automatically runs a post processing script on the output creating
    plots of the posterior and saving basic information like the evidence and
    run settings on a text file for future reference.

    Parameters
    ----------
    model : object
        Custom model class that contains your desired model. This class has to
        have a method called log_likelihood(x) which takes a parameter array x
        and returns the corresponding log Likelihood. The order of the array x is
        given by the order that results from calling list(priordict.keys()) (see
        description for priordict for more information.)
        Your custom model class should inherit from either RVModel if it's a 
        radial velocities model or from BaseModel otherwise.
    rundict : dict
        Dictionary with basic information about the run itself. Keys it should 
        include are:
            target : Name of target or function to analyse
            runid : String to identify the specific model or configuration being
                    used
            comment (optional) : Optional comment for a third layer of identification
            prior_names (optional) : List with the names and ranges of the priors
            nplanets (optional) : Number of planets in the RV model. If not 
                                  provided, it will use the number of planets 
                                  present in the configfile.
            star_params (optional) : Dictionary with the stars parameters like
                                     star_mass (mass of the star in solar masses),
                                     star_radius (radius of star in solar radii),
                                     star_rot (roation period of star). These
                                     can be be ufloats to include the uncertainty.
            savedir (optional) : Save directory for the output of PolyChord
            order_planets (optionsl) : Boolean indicating if the planets should
                                       be ordered by period. This is to avoid
                                       periods jumping between the different 
                                       planets. Default is False.
    priordict : dict
        Dictionary with the priors to all free parameters. Keys are the names
        of the parameters. Values are object with a method .ppf(x) which is the 
        inverse of the CDF of the chosen probability distribution of the prior.
        It takes a uniformly sampled number between 0 and 1 and returns the 
        physical parameter distributed according to the prior. 
        The method log_likelihood in your custom model should take the same order
        or parameters that results from calling list(priordict.keys()).
    polysettings : dict, optional
        Dictionary containing custom parameters for PolyChord setting like nlive
        or nrepeats. If None are given the defualt PolyChord settings.

    Returns
    -------
    output : PolyChordOutput object
        Object with the PolyChord output. 
        Several attributes are added before returning. These are used for the 
        post processing script.
    """

    # Create list of parameter names
    parnames = model.parnames
    rundict_keys = list(rundict.keys())

    # Count "real" number of planets if not provided by rundict or model
    nplanets = 0
    if 'nplanets' in rundict_keys:
        # Check if nplanets is in rundict
        nplanets = rundict['nplanets']
    elif hasattr(model, 'nplanets'):
        # Check if nplanets is in the model
        nplanets = model.nplanets
    else:
        # Count number of planets by checking for periods
        for i, par in enumerate(parnames):
            if ('planet' in par) and ('period' in par):
                nplanets += 1

    planets = []
    planet_idxs = []
    for n in range(1, nplanets+1):
        planets.append([])
        for i, par in enumerate(parnames):
            if f'planet{n}' in par:
                planets[n-1].append(i)
                if 'period' in par:
                    planet_idxs.append(i)

    # Prepare run
    nderived = 0
    ndim = len(parnames)

    # Function to convert from hypercube to physical parameter space
    def prior(hypercube):
        """
        Converts a point in the unit hypercube to the physical parameters using
        their respective priors.
        """

        # Claculate physical parameters with ppf from prior
        theta = np.ones_like(hypercube)
        for i, x in enumerate(range(ndim)):
            param = parnames[i]
            theta[x] = priordict[param].ppf(hypercube[i])

        return theta

    # LogLikelihood

    def loglike(x):
        """
        Calculates de logarithm of the Likelihood given the parameter vector x. 
        """

        return model.log_likelihood(x)

    # Starting time to identify this specific run
    # If it's being run with more than one core the isodate on the first core
    # is broadcasted to the rest so they all share the same variable
    isodate = datetime.datetime.today().isoformat()
    if size > 1:
        isodate = comm.bcast(isodate, root=0)

    # Create PolyChordSettings object for this run
    settings = set_ultrasettings(rundict, ultrasettings, ndim, nderived, isodate, parnames)

    # Find and indicate wrapped (circular) parameters like phases
    wrapped_params = np.zeros(ndim, dtype=bool)
    for i, par in enumerate(parnames):
        if 'omega' in par or 'ml0' in par:
            wrapped_params[i] = True

    # Set up the sampler
    sampler = ReactiveNestedSampler(parnames, loglike, prior,
                                    log_dir = settings['log_dir'],
                                    num_test_samples = 100,
                                    wrapped_params = wrapped_params,
                                    num_bootstraps = settings['num_bootstraps'],
                                     #resume=False,
                                    # vectorized = True
                                    )

    # Use a slice sampler instead of rejection sampling
    sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=settings['nsteps'], adaptive_nsteps='move-distance')

    # Initialise clocks
    ti = time.process_time()

    # ----- Run UltraNest ------
    sampler.run(min_num_live_points = settings['nlive'],
                cluster_num_live_points = int(0.1*settings['nlive']),
                dlogz = settings['dlogz'],
                frac_remain = settings['frac_remain']
                )
    # --------------------------

    # Stop clocks
    tf = time.process_time()

    if size > 1:
        # Reduce clocks to min and max to get actual wall time
        ti = comm.reduce(ti, op=MPI.MIN, root=0)
        tf = comm.reduce(tf, op=MPI.MAX, root=0)

    # Print General results
    sampler.print_results()

    # Save results
    if rank == 0:
        output = Output()
        # Assign additional parameters to output
        output.runtime = datetime.timedelta(seconds=tf-ti)
        output.rundict = rundict.copy()
        output.datadict = dict(model.datadict)
        output.fixedpardict = dict(model.fixedpardict)
        output.model_name = str(model.model_path.stem)
        output.nlive = settings['nlive']
        output.nrepeats = settings['nsteps']
        output.isodate = isodate
        output.ncores = size
        output.parnames = parnames
        output.ndim = ndim
        output.sampler = 'UltraNest'
        output.base_dir = settings['log_dir']
        output.file_root = settings['file_root']
        output.logZ = sampler.results['logz']
        output.logZerr = sampler.results['logzerr']
        output.nlike = sampler.results['ncall']

        output.samples = pd.DataFrame(sampler.results['samples'], columns=parnames)

        # Add additional information if provided
        if 'prior_names' in rundict_keys:
            output.priors = rundict['prior_names']

        if 'star_params' in rundict_keys:
            output.starparams = rundict['star_params']

        # Print run time
        print(f'\nTotal run time was: {output.runtime}')


        # Plot results
        sampler.plot()

        base_dir_parent = str(Path(output.base_dir).parent.absolute())
        runid_dir = Path(output.base_dir).parent.parent.absolute()
        # Save model file
        shutil.copy(model.model_path, base_dir_parent)
        # Save output as pickle file
        # print(base_dir_parent)
        # run_label = os.path.basename(base_dir_parent)
        # print(run_label)
        dump2pickle_poly(output, output.file_root+'.pkl')

        # Copy post processing script to this run's folder
        parent = Path(__file__).parent.parent.absolute()
        shutil.copy(os.path.join(parent, 'post_processing.py'), base_dir_parent)
        # Copy FIP criterion scirpt to parent of runid
        shutil.copy(os.path.join(parent, 'fip_criterion.py'), runid_dir)

        # Copy model file
        shutil.copy(model.model_path, base_dir_parent)

        # Run post processing script
        postprocess(base_dir_parent)

    return #output


def dump2pickle_poly(output, filename, savedir=None):
    """ 
    Takes the output from UltraNest and saves it as a pickle file.

    Parameters
    ----------
    output : Output object
        Object file with the output infos from the UltraNest run
    filename : str
        Name of the saved file
    savedir : str, optional
        Directory on where to save the pickled file
    """

    # Try to import pickle. Raise warning if not installed.
    try:
        import pickle
    except ImportError:
        print('Install pickle to save the output. Try running:\n pip install pickle')
        return

    if savedir is None:
        # Save directory in parent of base dir
        pickledir = Path(output.base_dir).parent
    else:
        # Unless specified otherwhise
        pickledir = savedir

    # Create directory if it doesn't exist.
    os.makedirs(pickledir, exist_ok=True)

    full_path = os.path.join(pickledir, filename)
    with open(full_path, 'wb') as f:
        pickle.dump(output, f)

    return


def set_ultrasettings(rundict, ultrasettings, ndim, nderived, isodate, parnames):
    """ 
    Sets the correct settings for UltraNest and returns a dictionary with the
    settings. It combines the default settings and overwrites the used defined
    settings. It cointains the settings for both the initialization of the
    samples and the its run function.

    Parameters
    ----------
    rundict : dict
        Dictionary with information about the run itself.
    polysettings : dict
        Dictionary with the custom UltraNest settings to be set for this run
    ndim : int
        Number of free parameters
    nderived : int
        Number of derived parameters
    isodate : datetime
        Date stamp to identify the current run
    parnames : list
        List containing the names of the free parameters

    Returns
    -------
    settings : dict
        Object with all the information that UltraNest needs to run nested
        sampling on this model.
    """

    rundict_keys = list(rundict.keys())
    # Define PolyChord settings
    # Use the settings provided in polysettings otherwise use default
    # Definition of default values for PolyChordSettings
    default_settings = {'nlive': 25*ndim,
                        'nsteps': 3*ndim,
                        'dlogz': 0.5,
                        'frac_remain': 0.01,
                        'num_bootstraps': 30
                        }

    # Update default values with settings provided by user
    if ultrasettings != None:
        if type(ultrasettings) is not dict:
            raise TypeError("ultrasettings has to be a dictionary")
        else:
            default_settings.update(ultrasettings)

    # Define fileroot name (identifies this specific run)
    rundict['target'] = rundict['target'].replace(' ', '')  # Remove any whitespace
    rundict['runid'] = rundict['runid'].replace(' ', '')  # Remove any whitespace
    file_root = rundict['target']+'_'+rundict['runid']

    # Add comment if it exists and is not empty
    if 'comment' in rundict_keys:
        if rundict['comment'] != '':
            file_root += '-' + rundict['comment']

    # Add number of planets if it exists
    if 'nplanets' in rundict_keys:
        if rundict['nplanets'] is not None:
            file_root += f'_k{rundict["nplanets"]}'

    # Check if there is a drift and add "d{n}" to file name
    drift_order = 0
    for par in parnames:
        if 'drift' in par:
            if par[6:] in ['lin', 'quad', 'cub', 'quar']:
                drift_order += 1
    if drift_order > 0:
        file_root += f'_d{drift_order}'

    # Label the run with nr of planets, live points, nr of cores, sampler and date
    file_root += f'_nlive{default_settings["nlive"]}'
    file_root += f'_ncores{size}'
    file_root += '_ultranest'
    file_root += '_'+isodate

    # Base directory
    # Check if a save directory was provided
    if 'save_dir' in rundict_keys:
        save_dir = rundict['save_dir']
    else:
        save_dir = ''
    base_dir = os.path.join(save_dir, file_root, 'ultraresults')

    # Update settings dictionary with fileroot and basedir
    default_settings.update({'log_dir': base_dir, 'file_root': file_root})

    # Create PolyChorSettings object and assign settings
    # settings = PolyChordSettings(ndim, nderived, **default_settings)
    # TODO Think about how to implement resumes

    return default_settings

class Output:
    pass
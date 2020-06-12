import os
import sys
import importlib
import datetime
import time
import shutil
import numpy as np
from pathlib import Path

from .post_processing import postprocess

# PolyChord imports
try:
    from pypolychord import run_polychord
    from pypolychord.settings import PolyChordSettings
except ImportError:
    raise ImportError("Install PolyChord to use this module.")

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


def run(model, rundict, priordict, polysettings=None):
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

    # Function to convert from hypercube to physical parameter space
    def prior(hypercube):
        """
        Converts a point in the unit hypercube to the physical parameters using
        their respective priors.
        """

        # Check if they are already sorted and skip ordering
        idxs = np.arange(len(hypercube), dtype=np.int)
        if 'order_planets' in rundict_keys:
            if rundict['order_planets']: 
                if not np.all(periods[:-1] <= periods[1:]):
                    # Get periods for sorting
                    periods = hypercube[planet_idxs]
                    # Sort periods
                    sorted_periods_args = np.argsort(periods)
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

        # Claculate physical parameters with ppf from prior
        theta = np.ones_like(hypercube)
        for i, x in enumerate(idxs):
            param = parnames[i]
            theta[x] = priordict[param].ppf(hypercube[i])

        return theta

    # LogLikelihood

    def loglike(x):
        """
        Calculates de logarithm of the Likelihood given the parameter vector x. 
        """

        return (model.log_likelihood(x), [])

    # Prepare run
    nderived = 0
    ndim = len(parnames)

    # Starting time to identify this specific run
    # If it's being run with more than one core the isodate on the first core
    # is broadcasted to the rest so they all share the same variable
    isodate = datetime.datetime.today().isoformat()
    if size > 1:
        isodate = comm.bcast(isodate, root=0)

    # Create PolyChordSettings object for this run
    settings = set_polysettings(
        rundict, polysettings, ndim, nderived, isodate, parnames)

    # Initialise clocks
    ti = time.process_time()

    # ----- Run PolyChord ------
    output = run_polychord(loglike, ndim, nderived, settings, prior)
    # --------------------------

    # Stop clocks
    tf = time.process_time()

    if size > 1:
        # Reduce clocks to min and max to get actual wall time
        ti = comm.reduce(ti, op=MPI.MIN, root=0)
        tf = comm.reduce(tf, op=MPI.MAX, root=0)

    # Save results
    if rank == 0:
        # Cleanup of parameter names
        paramnames = [(x, x) for x in parnames]
        output.make_paramnames_files(paramnames)
        # Delete loglike and weight columns
        # del output.samples['loglike']
        # del output.samples['weight']
        # old_cols = output.samples.columns.values.tolist()
        # output.samples.rename(columns=dict(
        #     zip(old_cols, parnames)), inplace=True)

        # Assign additional parameters to output
        output.runtime = datetime.timedelta(seconds=tf-ti)
        output.rundict = rundict.copy()
        output.datadict = dict(model.datadict)
        output.fixedpardict = dict(model.fixedpardict)
        output.model_name = str(model.model_path.stem)
        output.nlive = settings.nlive
        output.nrepeats = settings.num_repeats
        output.isodate = isodate
        output.ncores = size
        output.parnames = parnames
        output.ndim = ndim

        # Add additional information if provided
        if 'prior_names' in rundict_keys:
            output.priors = rundict['prior_names']

        if 'star_params' in rundict_keys:
            output.starparams = rundict['star_params']

        # Print run time
        print('\nTotal run time was: {}'.format(output.runtime))

        # Save output as pickle file
        dump2pickle_poly(output, output.file_root+'.dat')

        base_dir_parent = str(Path(output.base_dir).parent.absolute())
        # Save model as pickle file
        shutil.copy(model.model_path, base_dir_parent)
        # dump2pickle_poly(model, 'model.pkl', savedir=base_dir_parent)

        # Copy post processing script to this run's folder
        parent = Path(__file__).parent.absolute()
        shutil.copy(os.path.join(
            parent, 'post_processing.py'), base_dir_parent)

        # Copy model file
        shutil.copy(model.model_path, base_dir_parent)

        # Run post processing script
        postprocess(base_dir_parent)

    return output


def dump2pickle_poly(output, filename, savedir=None):
    """ 
    Takes the output from PolyChord and saves it as a pickle file.

    Parameters
    ----------
    output : PolyChordOutput object
        Object file with the output from the PolyChord run
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


def set_polysettings(rundict, polysettings, ndim, nderived, isodate, parnames):
    """ 
    Sets the correct settings for polychord and returns the PolyChordSettings 
    object.

    Parameters
    ----------
    rundict : dict
        Dictionary with information about the run itself.
    polysettings : dict
        Dictionary with the custom PolyChord settings to be set for this run
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
    settings : PolyChordSettings object
        Object with all the information that PolyChord needs to run nested
        sampling on this model.
    """

    rundict_keys = list(rundict.keys())
    # Define PolyChord settings
    # Use the settings provided in polysettings otherwise use default
    # Definition of default values for PolyChordSettings
    default_settings = {'nlive': 25*ndim,
                        'num_repeats': 5*ndim,
                        'do_clustering': True,
                        'write_resume': False,
                        'read_resume': False,
                        'feedback': 1,
                        'precision_criterion': 0.001,
                        'boost_posterior': 0.0
                        }

    # Update default values with settings provided by user
    if polysettings != None:
        if type(polysettings) is not dict:
            raise TypeError("polysettings has to be a dictionary")
        else:
            setting = 'nlive'
            if setting in polysettings.keys():
                if type(polysettings[setting]) is not int:
                    raise TypeError(
                        f'{setting} has to be an integer (got type {type(polysettings[setting])})')

            setting = 'num_repeats'
            if setting in polysettings.keys():
                if type(polysettings[setting]) is not int:
                    raise TypeError(
                        f'{setting} has to be an integer (got type {type(polysettings[setting])})')

            setting = 'do_clustering'
            if setting in polysettings.keys():
                if type(polysettings[setting]) is not bool:
                    raise TypeError(
                        f'{setting} has to be a boolean (got type {type(polysettings[setting])})')

            setting = 'read_resume'
            if setting in polysettings.keys():
                if type(polysettings[setting]) is not bool:
                    raise TypeError(
                        f'{setting} has to be a boolean (got type {type(polysettings[setting])})')

            setting = 'precision_criterion'
            if setting in polysettings.keys():
                if type(polysettings[setting]) is not float:
                    raise TypeError(
                        f'{setting} has to be a float (got type {type(polysettings[setting])})')

            default_settings.update(polysettings)

    # Define fileroot name (identifies this specific run)
    rundict['target'] = rundict['target'].replace(
        ' ', '')  # Remove any whitespace
    rundict['runid'] = rundict['runid'].replace(
        ' ', '')  # Remove any whitespace
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
    file_root += '_nlive{}'.format(default_settings['nlive'])
    file_root += '_ncores{}'.format(size)
    file_root += '_polychord'
    file_root += '_'+isodate

    # Base directory
    # Check if a save directory was provided
    if 'save_dir' in rundict_keys:
        save_dir = rundict['save_dir']
    else:
        save_dir = ''
    base_dir = os.path.join(save_dir, file_root, 'polychains')

    # Update settings dictionary with fileroot and basedir
    default_settings.update({'file_root': file_root, 'base_dir': base_dir})

    # Create PolyChorSettings object and assign settings
    settings = PolyChordSettings(ndim, nderived, **default_settings)
    # settings.nlive = default_settings['nlive']
    # settings.num_repeats = default_settings['num_repeats']
    # settings.do_clustering = default_settings['do_clustering']
    # # TODO Think about how to implement resumes
    # settings.write_resume = default_settings['write_resume']
    # settings.read_resume = default_settings['read_resume']
    # settings.feedback = default_settings['feedback']
    # settings.precision_criterion = default_settings['precision_criterion']
    # settings.file_root = default_settings['file_root']
    # settings.base_dir = default_settings['base_dir']
    # settings.boost_posterior = default_settings['boost_posterior']

    return settings

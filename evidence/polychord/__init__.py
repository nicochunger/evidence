import os
import sys
import numpy as np
import importlib
import datetime
import time
import shutil
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
    TODO Write detailed documentation of how to run polychord with your own model.
    """

    # Create list of parameter names
    parnames = list(priordict.keys())
    rundict_keys = list(rundict.keys())

    # Function to convert from hypercube to physical parameter space
    def prior(hypercube):
        """ 
        Convert a point in the unit hypercube to the physical parameters using
        their respective priors. 
        """

        theta = []
        for i, x in enumerate(hypercube):
            theta.append(priordict[parnames[i]].ppf(x))
        return theta    

    
    # LogLikelihood
    def loglike(x):
        """ 
        Calculates de logarithm of the Likelihood given the parameter vector x. 
        """

        loglike.nloglike += 1 # Add one to the likelihood calculations counter
        return (model.log_likelihood(x), [])
    loglike.nloglike = 0 # Likelihood calculations counter

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
    settings = set_polysettings(rundict, polysettings, ndim, nderived, isodate, parnames)

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

        # Gather all the number of likelihood calculations and sum to get total
        nlog = comm.reduce(loglike.nloglike, op=MPI.SUM, root=0)
    else:
        # If only one core was used just use the nloglike attribute
        nlog = loglike.nloglike

    # Save results
    if rank == 0:
        # Cleanup of parameter names
        paramnames = [(x, x) for x in parnames]
        output.make_paramnames_files(paramnames)
        # Delete loglike and weight columns
        del output.samples['loglike']
        del output.samples['weight']
        old_cols = output.samples.columns.values.tolist()
        output.samples.rename(columns=dict(zip(old_cols, parnames)), inplace=True)

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
        output.nloglike = nlog

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
        shutil.copy(os.path.join(parent,'post_processing.py'), base_dir_parent)

        # Copy model file
        shutil.copy(model.model_path, base_dir_parent)

        # Run post processing script
        postprocess(base_dir_parent)


    return output


def dump2pickle_poly(output, filename, savedir=None):
    """ Takes the output from PolyChord and saves it as a pickle file. """

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
    Sets the correct settings for polychord and returns
    the PolyChordSettings object.
    """

    rundict_keys = list(rundict.keys())
    # Define PolyChord settings
    # Use the settings provided in polysettings otherwise use default
    # Definition of default values for PolyChordSettings
    default_settings = {'nlive': 25*ndim,
                        'num_repeats': 5*ndim,
                        'do_clustering': True,
                        'read_resume': False,
                        'feedback': 1,
                        'precision_criterion': 0.01,
                        }

    # Update default values with settings provided by user
    if polysettings != None:
        if type(polysettings) is not dict:
            raise TypeError("polysettings has to be a dictionary")
        else:
            setting = 'nlive'
            if setting in polysettings.keys():
                if type(polysettings[setting]) is not int:
                    raise TypeError(f'{setting} has to be an integer (got type {type(polysettings[setting])})')

            setting = 'num_repeats'
            if setting in polysettings.keys():
                if type(polysettings[setting]) is not int:
                    raise TypeError(f'{setting} has to be an integer (got type {type(polysettings[setting])})')

            setting = 'do_clustering'
            if setting in polysettings.keys():
                if type(polysettings[setting]) is not bool:
                    raise TypeError(f'{setting} has to be a boolean (got type {type(polysettings[setting])})')

            setting = 'read_resume'
            if setting in polysettings.keys():
                if type(polysettings[setting]) is not bool:
                    raise TypeError(f'{setting} has to be a boolean (got type {type(polysettings[setting])})')

            setting = 'precision_criterion'
            if setting in polysettings.keys():
                if type(polysettings[setting]) is not float:
                    raise TypeError(f'{setting} has to be a float (got type {type(polysettings[setting])})')

            default_settings.update(polysettings)

    # Define fileroot name (identifies this specific run)
    rundict['target'] = rundict['target'].replace(' ', '') # Remove any whitespace
    rundict['runid'] = rundict['runid'].replace(' ', '') # Remove any whitespace
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
    settings = PolyChordSettings(ndim, nderived, )
    settings.nlive = default_settings['nlive']
    settings.num_repeats = default_settings['num_repeats']
    settings.do_clustering = default_settings['do_clustering']
    settings.read_resume = default_settings['read_resume'] # TODO Think about how to implement resumes
    settings.feedback = default_settings['feedback']
    settings.precision_criterion = default_settings['precision_criterion']
    settings.file_root = default_settings['file_root']
    settings.base_dir = default_settings['base_dir']

    return settings
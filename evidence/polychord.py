import os
import sys
import numpy as np
import importlib
import datetime
import time
import shutil

# PolyChord imports
try:
    import pypolychord as polychord
    import pypolychord.settings as polysettings
except ImportError:
    raise ImportError("Install PolyChord to use this module.")

# MPI imports
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    print("Install mpi4py to use with OpenMPI.")
    rank = 0
    size = 1

from .config import read_config

HOME = os.getenv('HOME')


def run(configfile, nlive=None, nplanets=None, modelargs={}, **kwargs):
    """ 
    TODO Write detailed documentation of how to run polychord with your own model.

    IDEA to generalize this function and not make it so particular to RV.
        Instead of inputing the configfile, the input should be the model itself
        and the rundict. It's only two things needed.
        Model should have a method called log_likelihood which takes the vector
        of parameters and outputs the loglikliehood.
        rundict should have three keys: 'target', 'runid' and 'comment'. These are used
        to give a name to the specific polychord run. They are meant to serve as
        layers of categorization. For example 'runid' can be used for different
        models for the same target, and 'comment' for different configurations or
        prior of the same model. Only 'target' and 'runid' are mandatory, 
        'comment' is optional.
        I will also need to add a 'save_path' key for people to indicate where they
        want to save the run.
    """


    # Read dictionaries from configuration file
    rundict, datadict, priordict, fixeddict, priors = read_config(configfile, nplanets)
    parnames = list(priordict.keys())

    # Import model module
    models_path = os.path.join(HOME, 'run/targets/{target}/models/{runid}'.format(**rundict))
    modulename = 'model_{target}_{runid}'.format(**rundict)
    sys.path.insert(0, models_path)
    mod = importlib.import_module(modulename) # modulename, models_path)

    # Instantiate model class (pass additional arguments)
    mymodel = mod.Model(fixeddict, datadict, parnames, **modelargs)

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
        return (mymodel.lnlike(x), [])
    loglike.nloglike = 0 # Likelihood calculations counter

    # Prepare run
    nderived = 0
    ndim = len(parnames)

    # Starting time to identify this specific run
    # Define it only for rank 0 and broadcoast to the rest
    if size > 1:
        # Do the broadcasting only if it's being run with more than one core
        if rank == 0:
            isodate = datetime.datetime.today().isoformat()
        else:
            isodate = None
        isodate = comm.bcast(isodate, root=0)
    else:
        isodate = datetime.datetime.today().isoformat()

    # Define PolyChord settings
    settings = polysettings.PolyChordSettings(ndim, nderived, )
    settings.do_clustering = True
    if nlive is None:
        settings.nlive = 25*ndim
    else:
        settings.nlive = nlive*ndim

    # Define fileroot name (identifies this specific run)
    fileroot = rundict['target']+'_'+rundict['runid']
    if rundict['comment'] != '':
        fileroot += '-' + rundict['comment']

    # Label the run with nr of planets, live points, nr of cores, sampler and date
    fileroot += '_k{}'.format(mymodel.nplanets)
    fileroot += '_nlive{}'.format(settings.nlive)
    fileroot += '_ncores{}'.format(size)
    fileroot += '_polychord'
    fileroot += '_'+isodate

    settings.file_root = fileroot
    settings.read_resume = False
    settings.num_repeats = ndim * 5
    settings.feedback = 1
    settings.precision_criterion = 0.01
    # Base directory
    ref_dir = os.path.join('ExP', rundict['target'], rundict['runid'], fileroot, 'polychains')
    if 'spectro' in HOME:
        # If it's runing in cluster -> save in scratch folder
        base_dir = os.path.join('/scratch/nunger', ref_dir)
    else:
        # Running locally
        base_dir = os.path.join(HOME, ref_dir)
    settings.base_dir = base_dir

    # Initialise clocks
    ti = time.process_time()

    # ----- Run PolyChord ------
    output = polychord.run_polychord(loglike, ndim, nderived, settings, prior)
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
        parnames.insert(0, 'loglike')
        parnames.insert(0, 'weight')
        old_cols = output.samples.columns.values.tolist()
        output.samples.rename(columns=dict(zip(old_cols, parnames)), inplace=True)

        # Assign additional parameters to output
        output.runtime = datetime.timedelta(seconds=tf-ti)
        output.target = rundict['target']
        output.runid = rundict['runid']
        output.comment = rundict.get('comment', '')
        output.nplanets = mymodel.nplanets
        output.nlive = settings.nlive
        output.nrepeats = settings.num_repeats
        output.isodate = isodate
        output.ncores = size
        output.priors = priors
        output.datadict = datadict
        output.parnames = parnames
        output.fixeddict = fixeddict
        output.nloglike = nlog
        try:
            output.starparams = rundict['star_params']
        except:
            pass

        # Print run time
        print('\nTotal run time was: {}'.format(output.runtime))

        # Save output as pickle file
        dump2pickle_poly(output)

        # Copy post processing script to this run's folder
        shutil.copy(os.path.join(HOME,'run/post_processing.py'), os.path.join(output.base_dir, '..'))

        # Copy model file to this run's folder
        model = os.path.join(models_path, modulename+'.py')
        shutil.copy(model, os.path.join(output.base_dir, '..'))

    return output


def dump2pickle_poly(output, savedir=None):
    """ Takes the output from PolyChord and saves it as a pickle file. """

    try:
        import pickle
    except ImportError:
        print('Install pickle to save the output. Try running:\n pip install pickle')
        return

    if savedir is None:
        # Save directory in parent of base dir
        pickledir = os.path.join(output.base_dir, '..')
    else:
        # Unless specified otherwhise
        pickledir = savedir

    # Create directory if it doesn't exist.
    os.makedirs(pickledir, exist_ok=True)

    full_path = os.path.join(pickledir, output.file_root+'.dat')
    with open(full_path, 'wb') as f:
        pickle.dump(output, f)

    return
import sys
import os
import importlib
import pandas as pd
from .priors import prior_constructor


def read_config(configfile, nplanets=None):
    """
    Initialises a sampler object using parameters in config.

    :param string config: string indicating path to configuration file.
    """

    # Import configuration file as module
    configpath = os.path.split(configfile)[0]
    file_name = os.path.split(configfile)[1][:-3]
    sys.path.insert(0, configpath)
    c = importlib.import_module(file_name)

    # Make copy of all relavant dictionaries
    rundict, input_dict, datadict = map(dict.copy, c.configdicts)

    # Check how many planets need to be added to the model
    num_planet_dicts = 0
    for key in input_dict.keys():
        if 'planet' in key:
            num_planet_dicts += 1

    if rundict['nplanets'] != None:
        nplanets = rundict['nplanets']
        assert nplanets == num_planet_dicts, (f"Number of planets is {nplanets}, " 
                                              f"but there are {num_planet_dicts} "
                                               "planet dicts")
    elif nplanets == None:
        raise AssertionError("Number of planets is not defined.")
    else:
        # Create dicts for each planet
        fpdict = dict(input_dict['planet1'])
        del input_dict['planet1']
        for i in range(1, nplanets+1):
            try:
                # Try deleting it first to avoid repeats
                del input_dict['planet{}'.format(i)]
            except:
                pass
            input_dict.update({'planet{}'.format(i): dict(fpdict)})
        rundict['nplanets'] = nplanets

    # Create prior instances
    priordict = prior_constructor(input_dict, {})

    # Build list of parameter names and priors
    priors = read_priors(input_dict)

    # Read data from file(s)
    read_data(c.datadict)

    # Fixed parameters
    fixedpardict = get_fixedparvalues(input_dict)

    return rundict, datadict, priordict, fixedpardict, priors


def get_parnames(input_dict):
    parnames = []
    fixparnames = []
    for obj in input_dict:
        for par in input_dict[obj]:
            if input_dict[obj][par][1] > 0:
                parnames.append(obj+'_'+par)
            elif input_dict[obj][par][1] == 0:
                fixparnames.append(obj+'_'+par)
    return parnames, fixparnames


def get_fixedparvalues(input_dict):
    fpdict = {}
    for obj in input_dict:
        for par in input_dict[obj]:
            if input_dict[obj][par][1] == 0:
                fpdict[obj+'_'+par] = input_dict[obj][par][0]
    return fpdict


def read_data(datadict):
    for inst in datadict:
        # Try to get custom separator
        try:
            sep = datadict[inst]['sep']
        except KeyError:
            sep = '\t'

        # Read rdb file
        data = pd.read_csv(datadict[inst]['datafile'], sep=sep,
                           comment='#', skiprows=[1, ])
        datadict[inst]['data'] = data
    return


def read_priors(input_dict):
    """ Reads the input dict and return a dictionary with all parameters and
    their respective priors. """

    priors = {}
    
    # Iteration over all parameter objects
    for objkey in input_dict.keys():

        # Iteration over all parameters of a given object
        for parkey in input_dict[objkey]:

            parlist = input_dict[objkey][parkey]

            if not isinstance(parlist, list):
                continue

            # If parameter does not jump, or is marginalised skip this element
            if parlist[1] == 0:
                continue

            # Construct prior instance with information on dictionary
            priortype = parlist[2][0]
            pars = parlist[2][1:]
            
            priors[objkey+'_'+parkey] = f'{priortype}: {pars}'

    return priors
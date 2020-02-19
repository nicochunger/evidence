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
    rundict, inputdict, datadict = map(dict.copy, c.configdicts)

    # Check nplanets argument and add or remove planet dicts
    if nplanets != None:
        # Check that nplanets has a correct value
        if type(nplanets) is not int:
            raise TypeError("nplanets has to be an integer.")
        elif nplanets < 0:
            raise ValueError("nplanets has to be positive.")

        # Check how many planet dicts are present in the configfile
        num_planet_dicts = 0
        for key in inputdict.keys():
            if 'planet' in key:
                num_planet_dicts += 1

        # Check that nplanets doesn't exceed the number of planets dicts
        # EXCEPT if there is only 1
        if num_planet_dicts > 1 and nplanets > num_planet_dicts:
            raise ValueError("Not enough planet dictionaries for the requested number of planets.")

        # Dynamically add or remove planet dicts for the requested amount of planets
        if num_planet_dicts == 1:
            # If only one planet dict is present, this one is duplicated for the rest
            fpdict = dict(inputdict['planet1'])
            del inputdict['planet1']
            for n in range(1, nplanets+1):
                inputdict.update({f'planet{n}': dict(fpdict)})

        elif num_planet_dicts > 1:
            for n in range(nplanets+1, num_planet_dicts+1):
                del inputdict[f'planet{n}']

        # Add nplanets field to rundict
        rundict.update({'nplanets': nplanets})

    # Create prior instances
    priordict = prior_constructor(inputdict)

    # Build list of parameter names and priors
    read_priors(inputdict, rundict)

    # Read data from file(s)
    read_data(c.datadict)

    # TODO Add option to add polysettings to configfile

    # Fixed parameters
    fixedpardict = get_fixedparvalues(inputdict)

    return rundict, datadict, priordict, fixedpardict


def get_parnames(inputdict):
    parnames = []
    fixparnames = []
    for obj in inputdict:
        for par in inputdict[obj]:
            if inputdict[obj][par][1] > 0:
                parnames.append(obj+'_'+par)
            elif inputdict[obj][par][1] == 0:
                fixparnames.append(obj+'_'+par)
    return parnames, fixparnames


def get_fixedparvalues(inputdict):
    fpdict = {}
    for obj in inputdict:
        for par in inputdict[obj]:
            if inputdict[obj][par][1] == 0:
                fpdict[obj+'_'+par] = inputdict[obj][par][0]
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


def read_priors(inputdict, rundict):
    """ Reads the input dict and return a dictionary with all parameters and
    their respective priors. """

    priors = {}
    
    # Iteration over all parameter objects
    for objkey in inputdict.keys():

        # Iteration over all parameters of a given object
        for parkey in inputdict[objkey]:

            parlist = inputdict[objkey][parkey]

            if not isinstance(parlist, list):
                continue

            # If parameter does not jump or is marginalised, skip this element
            if parlist[1] == 0:
                continue

            # Construct prior instance with information on dictionary
            priortype = parlist[2][0]
            pars = parlist[2][1:]
            # Round parameters so that 2*pi is not that long
            for i in range(len(pars)):
                pars[i] = round(pars[i], 4)
            
            priors[objkey+'_'+parkey] = f'{priortype}: {pars}'

    rundict.update({'prior_names': priors})

    return
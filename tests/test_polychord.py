import os
import sys
import pytest
from pathlib import Path
import importlib
import datetime

from evidence import polychord
from evidence import config

def poly_setup(nplanets=None):
    """ Sets up a basic polychord run for testing """

    parent_path = os.path.join(Path(__file__).parent.absolute(), 'test_examples/51Peg')
    configfile = os.path.join(parent_path, 'config_51Peg_example.py')

    # Read dictionaries from configuration file
    rundict, datadict, priordict, fixeddict = config.read_config(configfile, nplanets)
    parnames = list(priordict.keys())

    # Import model module
    modulename = 'model_51Peg_example'
    sys.path.insert(0, parent_path)
    mod = importlib.import_module(modulename)

    # Instantiate model class (pass additional arguments)
    model = mod.Model(fixeddict, datadict, parnames)

    return model, rundict, priordict


def test_polysettings():
    """ Tests the function which defines the run settings for PolyChord """

    model, rundict, priordict = poly_setup(nplanets=1)

    ndim = len(priordict.keys())
    nderived = 0
    isodate = datetime.datetime.today().isoformat()

    polysettings = None

    settings = polychord.set_polysettings(rundict, polysettings, ndim, nderived, isodate)

    assert settings.nlive == 175
    assert settings.num_repeats == 35
    assert settings.do_clustering == True
    assert settings.precision_criterion == 0.01

    # Test if invalid data types are handled correctly
    polysettings = {'nlive': 43.7}
    with pytest.raises(TypeError):
        polychord.set_polysettings(rundict, polysettings, ndim, nderived, isodate)

    polysettings = {'num_repeats': ['5']}
    with pytest.raises(TypeError):
        polychord.set_polysettings(rundict, polysettings, ndim, nderived, isodate)

    polysettings = {'do_clustering': {'clustering': True}}
    with pytest.raises(TypeError):
        polychord.set_polysettings(rundict, polysettings, ndim, nderived, isodate)

    polysettings = {'read_resume': 74}
    with pytest.raises(TypeError):
        polychord.set_polysettings(rundict, polysettings, ndim, nderived, isodate)
    
    polysettings = {'precision_criterion': False}
    with pytest.raises(TypeError):
        polychord.set_polysettings(rundict, polysettings, ndim, nderived, isodate)

@pytest.mark.slow
def test_gaussian_1d():

    parent_path = os.path.join(Path(__file__).parent.absolute(), 'test_examples/gaussian')
    configfile = os.path.join(parent_path, 'config_gaussian_1d.py')

    # Read dictionaries from configuration file
    rundict, datadict, priordict, fixeddict = config.read_config(configfile)
    parnames = list(priordict.keys())

    # Import model module
    modulename = 'model_gaussian_example'
    sys.path.insert(0, parent_path)
    mod = importlib.import_module(modulename)

    # Instantiate model class (pass additional arguments)
    model = mod.Model(fixeddict, datadict, parnames)

    polysettings = {'nlive': 500}

    output = polychord.run(model, rundict, priordict, polysettings)

    # Check that evidence is correct
    assert abs(output.logZ + 2.0) < 0.3

    # Check that output pickle is generated
    output_path = os.path.join(output.base_dir, '..', output.file_root+'.dat')
    assert os.path.exists(output_path)

    # Check that post processing script is copied correctly
    post_path = os.path.join(output.base_dir, '../post_processing.py')
    assert os.path.exists(post_path)

    # # Check that model files are copied correctly
    model_path = os.path.join(output.base_dir, '../model_gaussian_example.py')
    model_pkl_path = os.path.join(output.base_dir, '../model.pkl')
    assert os.path.exists(model_path)
    assert os.path.exists(model_pkl_path)

@pytest.mark.slow
def test_gaussian_2d():
    """ Checks that multidimensional models work. """

    parent_path = os.path.join(Path(__file__).parent.absolute(), 'test_examples/gaussian')
    configfile = os.path.join(parent_path, 'config_gaussian_2d.py')

    # Read dictionaries from configuration file
    rundict, datadict, priordict, fixeddict = config.read_config(configfile)
    parnames = list(priordict.keys())

    # Import model module
    modulename = 'model_gaussian_example'
    sys.path.insert(0, parent_path)
    mod = importlib.import_module(modulename)

    # Instantiate model class (pass additional arguments)
    model = mod.Model(fixeddict, datadict, parnames)

    polysettings = {'nlive': 500}

    output = polychord.run(model, rundict, priordict, polysettings)

    print(output.logZ)

    assert abs(output.logZ + 4.15) < 0.5


# @pytest.mark.skip(reason="Too slow, coment out this mark to test")
# def test_51Peg_k1():
#     """ Test a basic example of 51Peg"""

#     model, rundict, priordict = poly_setup(nplanets=1)

#     # Settings for PolyChord
#     polysettings = {'nlive': 5}

#     output = polychord.run(model, rundict, priordict, polysettings)

#     # Think what can be tested reliably
#     assert output.rundict['nplanets'] == 1
#     assert output.starparams['star_mass'] == 1.11
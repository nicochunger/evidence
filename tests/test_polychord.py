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

    parent_path = os.path.join(Path(__file__).parent.absolute(), 'example_test')
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


@pytest.mark.slow
def test_51Peg_k0():
    """ Test a basic example of 51Peg with 0 planets"""

    model, rundict, priordict = poly_setup(nplanets=0)

    # Settings for PolyChord
    polysettings = None

    output = polychord.run(model, rundict, priordict, polysettings)

    # Think what can be tested reliably
    assert 1 == 1


@pytest.mark.slow
def test_51Peg_k1():
    """ Test a basic example of 51Peg with 0 planets"""

    model, rundict, priordict = poly_setup(nplanets=1)

    # Settings for PolyChord
    polysettings = {'nlive': 100}

    output = polychord.run(model, rundict, priordict, polysettings)

    # Think what can be tested reliably
    assert 1 == 1
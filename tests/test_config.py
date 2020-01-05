import os
import pytest
from pathlib import Path

from evidence import config

def test_nplanets_type():
    """ Test that read_config raises the correct erors when invalid data types
    are given. """

    parent_path = os.path.join(Path(__file__).parent.absolute(), 
                               'test_examples/51Peg')
    configfile = os.path.join(parent_path, 'config_51Peg_example.py')

    with pytest.raises(TypeError):
        config.read_config(configfile, nplanets=0.5)

    with pytest.raises(ValueError):
        config.read_config(configfile, nplanets=-6)

def test_no_planets():
    """ Test for the config module """

    parent_path = os.path.join(Path(__file__).parent.absolute(), 
                               'test_examples/51Peg')
    configfile = os.path.join(parent_path, 'config_51Peg_example.py')

    # Test basic read config functionalities (without planet information)
    rundict, datadict, priordict, fixeddict = config.read_config(configfile)
    parnames = list(priordict.keys())
    assert len(parnames) == 7
    assert len(fixeddict.keys()) == 1
    assert len(datadict['hamilton']['data']) == 256
    assert rundict['target'] == '51Peg'
    assert rundict['star_params']['star_mass'] == 1.11
    assert rundict['save_dir'] == 'chains'
    assert 'nplanets' not in rundict.keys()

def test_with_planets():
    """ Test for the config module """

    parent_path = os.path.join(Path(__file__).parent.absolute(), 
                               'test_examples/51Peg')
    configfile = os.path.join(parent_path, 'config_51Peg_example.py')

    # Test read config functionalities when adding planets
    rundict, datadict, priordict, fixeddict = config.read_config(configfile, nplanets=0)
    parnames = list(priordict.keys())
    assert rundict['nplanets'] == 0
    for par in parnames:
        assert 'planet' not in par
    
    # Test with 1 planet
    rundict, datadict, priordict, fixeddict = config.read_config(configfile, nplanets=1)
    parnames = list(priordict.keys())
    assert rundict['nplanets'] == 1
    assert 'planet1_period' in parnames
    # Check that there really is only one planet present
    nplanets = 0
    for par in parnames:
        if 'k1' in par:
            nplanets += 1
    assert nplanets == 1

    # Test with 2 planet
    rundict, datadict, priordict, fixeddict = config.read_config(configfile, nplanets=2)
    parnames = list(priordict.keys())
    assert rundict['nplanets'] == 2
    assert 'planet1_period' in parnames
    assert 'planet2_period' in parnames
    # Check that there really are two planet present
    nplanets = 0
    for par in parnames:
        if 'k1' in par:
            nplanets += 1
    assert nplanets == 2
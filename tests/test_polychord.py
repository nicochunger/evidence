import os
import sys
import subprocess
import shutil
import pytest
from pathlib import Path
import importlib
import datetime

from evidence import polychord, config

def poly_setup(configfile, nplanets=None):
    """ Sets up a basic polychord run for testing """

    parent_path = os.path.join(Path(__file__).parent.absolute(), 'test_examples/51Peg')
    configfile = os.path.join(parent_path, configfile)

    # Read dictionaries from configuration file
    rundict, datadict, priordict, fixedpardict = config.read_config(configfile, nplanets)
    parnames = list(priordict.keys())

    # Import model module
    modulename = 'model_51Peg_example'
    sys.path.insert(0, parent_path)
    mod = importlib.import_module(modulename)

    # Instantiate model class (pass additional arguments)
    model = mod.Model(fixedpardict, datadict, parnames)

    return model, rundict, priordict


def test_polysettings():
    """ Tests the function which defines the run settings for PolyChord """

    _, rundict, priordict = poly_setup('config_51Peg_example.py', nplanets=1)

    ndim = len(priordict.keys())
    nderived = 0
    isodate = datetime.datetime.today().isoformat()
    parnames = list(priordict.keys())

    polysettings = None

    settings = polychord.set_polysettings(rundict, polysettings, ndim, nderived, isodate, parnames)

    assert settings.nlive == 175
    assert settings.num_repeats == 35
    assert settings.do_clustering == True
    assert settings.precision_criterion == 0.001

    # Test if invalid data types are handled correctly
    polysettings = {'nlive': 43.7}
    with pytest.raises(TypeError):
        polychord.set_polysettings(rundict, polysettings, ndim, nderived, isodate, parnames)

    polysettings = {'num_repeats': ['5']}
    with pytest.raises(TypeError):
        polychord.set_polysettings(rundict, polysettings, ndim, nderived, isodate, parnames)

    polysettings = {'do_clustering': {'clustering': True}}
    with pytest.raises(TypeError):
        polychord.set_polysettings(rundict, polysettings, ndim, nderived, isodate, parnames)

    polysettings = {'read_resume': 74}
    with pytest.raises(TypeError):
        polychord.set_polysettings(rundict, polysettings, ndim, nderived, isodate, parnames)
    
    polysettings = {'precision_criterion': False}
    with pytest.raises(TypeError):
        polychord.set_polysettings(rundict, polysettings, ndim, nderived, isodate, parnames)

    return

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

    polysettings = {'nlive': 100}

    output = polychord.run(model, rundict, priordict, polysettings)

    # Check that evidence is correct
    assert abs(output.logZ + 2.0) < 0.5
    
    # Clean old runs, it keep the newest one so the checks should pass
    clean_runs()

    # Check that output files were generated
    base_dir_parent = Path(output.base_dir).parent.absolute()
    assert os.path.isfile(os.path.join(base_dir_parent, output.file_root+'.pkl'))
    assert os.path.isfile(os.path.join(base_dir_parent, 'post_processing.py'))
    assert os.path.isfile(os.path.join(base_dir_parent, 'results.txt'))
    assert os.path.isfile(os.path.join(base_dir_parent, 'model_gaussian_example.py'))

    return



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

    # Clean old runs, it keep the newest one so the checks should pass
    clean_runs()
    
    # Check that output files were generated
    base_dir_parent = Path(output.base_dir).parent.absolute()
    assert os.path.isfile(os.path.join(base_dir_parent, output.file_root+'.pkl'))
    assert os.path.isfile(os.path.join(base_dir_parent, 'post_processing.py'))
    assert os.path.isfile(os.path.join(base_dir_parent, 'results.txt'))
    assert os.path.isfile(os.path.join(base_dir_parent, 'model_gaussian_example.py'))

    return


# @pytest.mark.skip(reason="Too slow, coment out this mark to test")
def test_51Peg_k1_d0():
    """ Test a basic example of 51Peg"""

    model, rundict, priordict = poly_setup('config_51Peg_example.py', nplanets=1)

    # Settings for PolyChord
    polysettings = {'nlive': 5}

    output = polychord.run(model, rundict, priordict, polysettings)

    clean_runs()
    # For now it only checks that it runs succesfully
    assert output.rundict['nplanets'] == 1
    assert output.starparams['star_mass'][0] == 1.11
    # Check that postprocessing created the files
    assert os.path.isfile(os.path.join(output.base_dir, '../post_processing.py'))
    assert os.path.isfile(os.path.join(output.base_dir, '../results.txt'))
    assert os.path.isfile(os.path.join(output.base_dir, '../model_51Peg_example.py'))
    assert os.path.isfile(os.path.join(output.base_dir, '../'+output.file_root+'.pkl'))

    return

# @pytest.mark.skip(reason="Too slow, coment out this mark to test")
def test_51Peg_k1_d1():
    """ Test a basic example of 51Peg"""

    model, rundict, priordict = poly_setup('config_51Peg_drift.py', nplanets=1)

    # Settings for PolyChord
    polysettings = {'nlive': 10}
    # polysettings = None

    output = polychord.run(model, rundict, priordict, polysettings)

    # For now it only checks that it runs succesfully
    assert output.rundict['nplanets'] == 1
    assert output.starparams['star_mass'][0] == 1.11
    # Check that postprocessing created the files
    assert os.path.isfile(os.path.join(output.base_dir, '../post_processing.py'))
    assert os.path.isfile(os.path.join(output.base_dir, '../results.txt'))
    assert os.path.isfile(os.path.join(output.base_dir, '../model_51Peg_example.py'))
    assert os.path.isfile(os.path.join(output.base_dir, '../'+output.file_root+'.pkl'))

    clean_runs()
    return

def clean_runs():
    # Delete chain folders if there are too many (too keep it clean)
    chains_path = os.path.join(Path(__file__).parent.absolute(), 'chains/')
    # chains_path = os.path.join(os.getenv('HOME'), 'evidence/tests/chains/')
    runs = os.listdir(chains_path)
    mod_times = []
    for run in runs:
        mod_times.append(os.path.getmtime(os.path.join(chains_path, run)))

    # Sorted the runs by modification time and delete all but the last 5
    sorted_runs = [x for _,x in sorted(zip(mod_times,runs))]
    while len(sorted_runs) > 5:
        shutil.rmtree(os.path.join(chains_path, sorted_runs[0]))
        sorted_runs.pop(0)

    # Check that it worked
    assert len(os.listdir(chains_path)) == 5

    return
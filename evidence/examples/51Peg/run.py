import os
import sys
import importlib
from pathlib import Path
from evidence import config, polychord

configfile = 'config_51Peg_example.py'

parent_path = Path(__file__).parent.absolute()
configfile = os.path.join(parent_path, configfile)

nplanets = 1

# Read dictionaries from configuration file
rundict, datadict, priordict, fixedpardict = config.read_config(configfile, nplanets)
parnames = list(priordict.keys())

# Import model module
modulename = 'model_51Peg_example'
sys.path.insert(0, parent_path)
mod = importlib.import_module(modulename)

# Instantiate model class (pass additional arguments)
model = mod.Model(fixedpardict, datadict, parnames)

# Set PolyChord run settings
polysettings = {'nlive': 5}

output = polychord.run(model, rundict, priordict, polysettings)

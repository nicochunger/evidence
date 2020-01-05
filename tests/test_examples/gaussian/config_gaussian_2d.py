# Model for the star HD40307
import numpy as np
import os
from pathlib import Path

# -------------------
# RUN INFO
# -------------------
rundict = {
    'target': 'gaussian',
    'runid': '1d',
    'save_dir': 'chains'
}

# -------------------
# DATA
# -------------------
# FIXME name of instrument in datadict and input_dict HAVE to be the same
# Find a way to do this automatically so there is no messup
datadict = {}

# ---------------------
# PARAMETERS
# --------------------
pardict = {'x1': [0., 1, ['Uniform', -10, 10]],
           'x2': [0., 1, ['Uniform', -10, 10]]}

input_dict = {'par': pardict}

# Build config dict
configdicts = [rundict, input_dict, datadict]

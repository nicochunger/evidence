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
    'save_dir': 'tests/chains'
}

# -------------------
# DATA
# -------------------
# FIXME name of instrument in datadict and input_dict HAVE to be the same
# Find a way to do this automatically so there is no messup
datadict = {
    # 'hamilton':
    # {
    #     'datafile':  os.path.join(Path(__file__).parent.absolute(), '51Peg.rv'),
    #     'sep': '\t',
    #     'instrument': 'hamilton'
    # }
}

# ---------------------
# PARAMETERS
# --------------------
pardict = {'x1': [0., 1, ['Uniform', -10, 10]]}

input_dict = {'par': pardict}

# Build config dict
configdicts = [rundict, input_dict, datadict]

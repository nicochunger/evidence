import os
from pathlib import Path

# -------------------
# RUN INFO
# -------------------
rundict = {
    'target': 'gaussian',
    'runid': '2d',
    'save_dir': os.path.join(Path(__file__).parent.parent.parent.absolute(), 'chains')
}

# -------------------
# DATA
# -------------------
datadict = {}

# ---------------------
# PARAMETERS
# --------------------
pardict = {'x1': [0., 1, ['Uniform', -10, 10]],
           'x2': [0., 1, ['Uniform', -10, 10]]}

input_dict = {'par': pardict}

# Build config dict
configdicts = [rundict, input_dict, datadict]

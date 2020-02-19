# Model for the star HD40307
import numpy as np
import os
from pathlib import Path

# -------------------
# RUN INFO
# -------------------
rundict = {
    'target': '51Peg',
    'runid': 'example',
    #'comment': '',
    'star_params': 
    {
        #'star_rot': star_rot, # Rotation period of target in days
        'star_mass': 1.11, # Mass of target in solar masses
        # 'star_radius': star_radius # Radius of target in solar radii
    },
    'save_dir': os.path.join(Path(__file__).parent.parent.parent.absolute(), 'chains')
}

# -------------------
# DATA
# -------------------
# FIXME name of instrument in datadict and input_dict HAVE to be the same
# Find a way to do this automatically to avoid problems
datadict = {
    'hamilton':
    {
        'datafile':  os.path.join(Path(__file__).parent.absolute(), '51Peg.rv'),
        'sep': '\t',
        'instrument': 'hamilton'
    }
}

# ---------------------
# PARAMETERS
# --------------------
planetdict1 = {'k1': [0.0, 1, ['Uniform', 0., 200.]],
               'period': [0.0, 1, ['UniformFrequency', 1, 1e3]],
               'ecc': [0.1, 1, ['Beta', 0.867, 3.03]],
               'omega': [0.1, 1, ['Uniform', 0., 2*np.pi]],
               'ma0': [0.1, 1, ['Uniform', 0., 2*np.pi]],
               'epoch': [51050, 0]
               }

hamiltondict = {'offset': [0., 1, ['Uniform', -10, 10]],
                'jitter': [0.75, 1, ['Uniform', 0., 50.]]
               }

driftdict = {'lin': [0., 1, ['Uniform', -100, 100]],
             'tref': [51050, 0.],
             'reloco': [345345, 0]
             }

input_dict = {'planet1': planetdict1,
              'hamilton': hamiltondict,
              'drift': driftdict,
              }

# Build config dict
configdicts = [rundict, input_dict, datadict]

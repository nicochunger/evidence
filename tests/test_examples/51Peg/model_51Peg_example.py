import numpy as np
import pandas as pd
from pathlib import Path
from evidence.rvmodel import RVModel

# Definition of Model class
class Model(RVModel):
    """
    The model.
    """
    
    def __init__(self, fixedpardict, datadict, parnames):
        super().__init__(fixedpardict, datadict, parnames)

        self.model_path = Path(__file__).absolute()
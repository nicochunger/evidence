import numpy as np
from pathlib import Path
from evidence.rvmodel import BaseModel

# Definition of Model class
class Model(BaseModel):
    """
    The model.
    """  

    def __init__(self, fixedpardict, datadict, parnames):
        super().__init__(fixedpardict, datadict, parnames)

        self.model_path = Path(__file__).absolute()

    def log_likelihood(self, x):
        """
        Compute log likelihood for parameter vector x
        
        :param array x: parameter vector, given in order of parnames
        attribute.
        """

        sigma = 1
        mu = 0
        loglike = - (0.5/sigma**2)*np.sum((x-mu)**2)

        return loglike
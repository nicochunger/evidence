import os
import numpy as np
import pandas as pd
from pathlib import Path
# For running C functions
from ctypes import cdll, c_double, c_int, POINTER, c_float
# Spleaf package for covariance matrix
from spleaf.rv import Cov

# Definition of Model class
class RVModel(object):
    """
    The model.
    """
    
    def __init__(self, fixedpardict, datadict, parnames):
        
        # Define self variables
        self.fixedpardict = fixedpardict
        self.parnames = parnames
        self.datadict = datadict
        # TODO Think how this will work with this template model
        # The other file will hace to be copied but this one can stay here
        # because it is installed as part of the evidence package
        # self.model_path = Path(__file__).absolute()


        # Prepare true anomaly C function
        self.lib = cdll.LoadLibrary('trueanomaly.so')
        self.lib.trueanomaly.argtypes = [POINTER(c_double), c_int, c_double,
                                         POINTER(c_double), c_int, c_double]

        # Count number of planets in model
        self.nplanets = 0
        for par in self.parnames:
            if 'k1' in par:
                self.nplanets += 1

        # Save number of instruments
        self.insts = list(datadict.keys())

        # Construct unified data array
        self.data = pd.DataFrame()
        for i, instrument in enumerate(list(datadict.keys())):
            datadict[instrument]['data']['inst_id'] = np.zeros(len(datadict[instrument]['data']), dtype=np.int) + i
            self.data = pd.concat([self.data,datadict[instrument]['data']], ignore_index=True)
        
        return


        
    def predict_rv(self, pardict, time, exclude_planet=None):
        """
        Give rv prediction of model at time t. Has the option to exlude the
        contribution of one of the planets. This option is used for phase folds.
        Leave it at None if all planets should be included.
        
        Parameters
        ----------
        pardict : dict
            Dictionary with parameters for which prediction is to be computed.
        time : ndarray or float
            Time(s) at which prediction is computed
        exclude_planet : int, optional
            Number of the planet to be exluded

        Returns
        -------
        rv_prediction : ndarray or float
            Combined predicted radial velocities for the given times and orbital
            parameters of all planets (except the exluded planet)
        """

        assert (type(exclude_planet) == None) or (type(exclude_planet) == int), \
                                        "exclude_planet has to be an integer."

        # Add fixed parameters to pardict
        pardict.update(self.fixedpardict)
    
        # Prepare array for planet-induced velocities
        rv_planet = np.zeros((self.nplanets, len(time)))

        for i in range(1, self.nplanets+1):
            # Skip planet if it is the one to be exluded
            if i != exclude_planet:
                rv_planet[i-1] = self.modelk(pardict, time, planet=i)
        
        # Sum effect of all planets to predicted velocity
        rv_prediction = rv_planet.sum(axis=0)

        return rv_prediction
  

    def log_likelihood(self, x):
        """
        Compute log likelihood for parameter vector x
        
        :param array x: parameter vector, given in order of parnames
        attribute.
        """

        pardict = {}
        for i, par in enumerate(self.parnames):
            pardict[par] = x[i]
        
        # Add fixed parameters to pardict
        pardict.update(self.fixedpardict)
        
        # Get time, rv, and correct for global offsets
        t = self.data['rjd'].values.copy()
        y = self.data['vrad'].values.copy()
        err = self.data['svrad'].values.copy()
        noise = np.zeros_like(err)
        for i, instrument in enumerate(self.insts):
            # Substract offsets
            y[np.where(self.data['inst_id']==i)] -= pardict[f'{instrument}_offset']
            # Add jitter to noise
            noise[np.where(self.data['inst_id']==i)] = err[np.where(self.data['inst_id']==i)]**2 + pardict[f'{instrument}_jitter']**2

        # RV prediction
        rvm = self.predict_rv(pardict, t)

        # Residual
        res = y - rvm

        loglike = self.loglikelihood(res, noise)

        return loglike


    def loglikelihood(self, residuals, noise):
        N = len(residuals)
        cte = -0.5 * N * np.log(2*np.pi)
        return cte - np.sum(np.log(np.sqrt(noise))) - np.sum(residuals**2 / (2 * noise))


    def modelk(self, pardict, time, planet):
        """
        Compute Keplerian curve of the radial velocity for a given planet.

        Parameters
        ----------
        pardict : dict
            Dictionary with the values for all parameters.
            Keplerian parameters (K, P, sqrt(e)*cos(w), sqrt(e)*sin(w), L0, v0, epoch)
            TODO Add all options that the parameters can take
        time : ndarray or float
            Time(s) for which to calculate the keplerian curve.
        planet : int
            Number of the planet for which to calculate its keplerian curve in 
            the radial velocities space

        Returns
        -------
        keplerian : ndarray
            Array with the predicted radial velocities for that specific planet 
            at the specidied times.
        """

        try:
            K_ms = pardict[f'planet{planet}_k1']
        except KeyError:
            K_ms = np.exp(pardict[f'planet{planet}_logk1'])

        try:
            P_day = pardict[f'planet{planet}_period']
        except KeyError:
            P_day = np.exp(pardict[f'planet{planet}_logperiod'])

        # Calculate eccentricity and mean anomaly at epoch 
        # according to chosen parameters
        # SESIN SECOS ML0
        if f'planet{planet}_secos' in pardict:
            secos = pardict[f'planet{planet}_secos']
            sesin = pardict[f'planet{planet}_sesin']
            ecc = secos**2 + sesin**2
            omega_rad = np.arctan2(sesin, secos)

        elif f'planet{planet}_ecos' in pardict:
            ecos = pardict[f'planet{planet}_ecos']
            esin = pardict[f'planet{planet}_esin']
            ecc = np.sqrt(ecos**2 + esin**2)
            omega_rad = np.arctan2(esin, ecos)

        else:
            try:
                ecc = pardict[f'planet{planet}_ecc']
                omega_rad = pardict[f'planet{planet}_omega']
            except KeyError:
                raise KeyError(
                    'Something is wrong with the eccentricity parametrisation')

        try:
            # Compute mean anomaly at epoch
            ml0 = pardict[f'planet{planet}_ml0']
            ma0_rad = ml0 - omega_rad
        except:
            ma0_rad = pardict[f'planet{planet}_ma0']

        epoch = pardict[f'planet{planet}_epoch']

        # Compute mean anomaly at current time (or array)
        ma = 2*np.pi/P_day * (time - epoch) + ma0_rad
        # Compute true anomaly
        nu = self.true_anomaly(ma, ecc)

        return K_ms * (np.cos(nu + omega_rad) + ecc * np.cos(omega_rad))


    def true_anomaly(self, ma, ecc):
        """ 
        Takes the mean anomaly and the eccentricity and computes the true
        anomaly. This is done using the Newton-Raphson method and is the most
        time consuming part of the entire likelihood calculation. That's why
        this function is written in C and called here from python. Doing this
        has shown to be much quicker than just writing it directly in python.

        Parameters
        ----------
        ma : ndarray
            Array with the values of the mean anomaly.
        ecc : float
            Eccentricity

        Returns
        -------
        nu : ndarray
            Array with the values for the true anomaly
        """

        # Pre allocate memory for the result
        nu = np.zeros_like(ma, dtype=c_double)
        
        self.lib.trueanomaly(ma.ctypes.data_as(POINTER(c_double)), len(ma), ecc,
                             nu.ctypes.data_as(POINTER(c_double)), int(1.0e4),
                             c_double(1.0e-4))

        return nu




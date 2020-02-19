import os
import numpy as np
import pandas as pd
import warnings
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
        """
        Initialize the base model class. This loads the data, counts the number of 
        planets in the model and provides some basic methods as a base for radial 
        velocities modelling. The main log_likelihood method should be implemented
        in the final model class which inherits this one.

        Parameters
        ----------
        fixedpardict : dict
            Dictionary with all the parameters of the model which are fixed and
            their value.
        datadict : dict
            Dictionary with all the information related to the data and the
            instruments.
        parnames : array_like
            List with the names of all free parameters
        """

        # Define self variables
        self.fixedpardict = fixedpardict
        self.parnames = parnames

        # Count number of planets in model
        self.nplanets = 0
        for par in self.parnames:
            if 'k1' in par:
                self.nplanets += 1

        # Prepare true anomaly C function
        self.lib = cdll.LoadLibrary(os.path.join(Path(__file__).parent.absolute(), 'trueanomaly.so'))
        self.lib.trueanomaly.argtypes = [POINTER(c_double), c_int, c_double,
                                         POINTER(c_double), c_int, c_double]

        # Save list of instruments
        self.insts = list(datadict.keys())

        # Construct unified data array and add column with instrument id
        self.datadict = datadict
        self.data = pd.DataFrame()
        for i, instrument in enumerate(self.insts):
            datadict[instrument]['data']['inst_id'] = np.zeros(len(datadict[instrument]['data']), dtype=np.int) + i
            self.data = pd.concat([self.data,datadict[instrument]['data']], ignore_index=True)
        
        return


        
    def predict_kep_rv(self, pardict, time, exclude_planet=None):
        """
        Give rv prediction for all planets at time t. Has the option to exlude the
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

        assert (type(exclude_planet) == type(None)) or (type(exclude_planet) == int), \
                f"exclude_planet has to be an {int}, got {type(exclude_planet)}."

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
        Compute log likelihood for parameter vector x.
        
        Parameters
        ----------
        x : array_like
            Parameter vector, given in order of parnames attribute.

        Returns
        -------
        log_likelihood : float
            Value of the log likelihood for parameter vector x
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
        rvm = self.predict_kep_rv(pardict, t)

        # Add drift (if there is one)
        rvm += self.drift(pardict, t)

        # Residual
        res = y - rvm

        loglike = self.logL(res, noise)

        return loglike


    def logL(self, residuals, noise):
        """
        Basic log likelihood function with gaussian white noise.

        Parameters
        ----------
        residuals : ndarray
            Residuals between the data and the model
        noise : ndarray
            Array for the noises of each data point. This should include all
            instrumental errors as well as any additional jitter noise.

        Returns
        -------
        loglikelihood : float
            Value of the log(Likelihood)
        """

        N = len(residuals) # Number of data points
        cte = -0.5 * N * np.log(2*np.pi)
        return cte - np.sum(np.log(np.sqrt(noise))) - np.sum(residuals**2 / (2 * noise))


    def drift(self, pardict, time):
        """
        Calculate the contribution of a global drift to the RV.
        Up to fourth order polynomials.

        Parameters
        ---------
        pardict : dict
            Dictionary with the values for all parameters. In this case it will
            look for ['drift_lin', 'drift_quad', 'drift_cub', 'drift_quar'].
            Units for these parameters should be m/s/yr^(order)
        time : ndarray
            Time(s) for which to calculate the drift.

        Returns
        -------
        drift : ndarray
            Predicted radial velocities for the global drift
        """

        parameters = list(pardict.keys())
        lin, quad, cub, quar = (0., 0., 0., 0.)

        # Search for drift parameters in the pardict
        if 'drift_lin' in parameters:
            lin = pardict['drift_lin']
        if 'drift_quad' in parameters:
            quad = pardict['drift_quad']
        if 'drift_cub' in parameters:
            cub = pardict['drift_cub']
        if 'drift_quar'in parameters:
            quar = pardict['drift_quar']

        # Shift time with given reference time and convert to years
        tref = 0
        if 'drift_tref' in parameters:
            tref = pardict['drift_tref']
        else:
            tref = time[0]

        # Check that there is no strange parameter in drift
        for par in parameters:
            if 'drift' in par:
                if par[6:] not in ['lin', 'quad', 'cub', 'quad', 'tref']:
                    warnings.warn(f"Found custom parameter '{par}' in drift which will"+\
                                    " not be taken into account unless it was"+\
                                    " specifically used in your model.")

        tt = (time - tref)/365.25
        drift = lin*tt + quad*tt**2 + cub*tt**3 + quar*tt**4

        return drift


    def linear_parameter(self, time, indicator, pardict, par, kernel='gaussian', timescale=0.5):
        """
        RV prediction for a linear dependece with some activity indicator. With
        option to smooth using a gaussian, box or epanechnikov kernel.

        Parameters
        ----------
        time : ndarray or float
            Time(s) for which to calculate the drift.
        indicator : ndarray
            Time series for the indicator that will be used as a linear dependence.
            Has to have the same length as time
        pardict : dict
            Dictionary with all the parameters and their values
        par : str
            Name of the parameters that is the scale for the linear parameter
        kernel : str, optional
            Kernel that will be used to smooth the series. This can be None if 
            no smoothing should be applied. Or the smoothing options are 'gaussian',
            'box', and 'epanechnikov'.
        timescale : float
            Time scale for the smoothing.

        Returns
        -------
        linear_scale : ndarray
            Array with the RV prediction for this linear dependence.
        """

        assert len(time) == len(indicator), "time and indicator have to have the \
                                             same length."
        # Smooth time series of chosen indicator
        renorm_time = time/(365.25 * timescale)
        series_smoothed = np.empty_like(indicator)
        for k in range(renorm_time.size):
            delta_t = renorm_time - renorm_time[k]
            if kernel == 'gaussian':
                w = np.exp(-0.5 * delta_t**2)
            elif kernel == 'box':
                w = np.abs(delta_t)<=1.0
            elif kernel == 'epanechnikov':
                w = (np.abs(delta_t)<=1.0)*(1.0-delta_t**2)
            else:
                raise ValueError("Chosen kernel is not a valid option.")

            # Normalize
            w /= np.sum(w)
            series_smoothed[k] = np.sum(w*indicator)

        # Normalize smoothed series to the interval [-1, 1]
        maxval = np.max(series_smoothed)
        minval = np.min(series_smoothed)
        norm_smooth = 2.0*(series_smoothed-minval)/(maxval-minval) - 1.0
        
        # Calculate predicted RV
        return pardict[par] * norm_smooth

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

        # Extract semi amplitude and period
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




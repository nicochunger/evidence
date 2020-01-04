import numpy as np
import pandas as pd
from pathlib import Path

# Definition of Model class
class Model(object):
    """
    The model.
    """
    
    def __init__(self, fixedpardict, datadict, parnames):
        
        # Define self variables
        self.fixedpardict = fixedpardict
        self.parnames = parnames
        self.datadict = datadict
        self.model_path = Path(__file__).absolute()

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


        
    def predict_rv(self, pardict, time, planet=None):
        """
        Give rv prediction of model at time t. Has the option to exlude the
        contribution of one of the planets. This option is used for phase folds.
        Leave it at None if all planets should be included.
        
        :param dict pardict: dictionary with parameters for which
        prediction is to be computed.
        :param nd.array or float time: time(s) at which prediction is
        computed
        :param int planet: number of planet to be exluded.
        """

        # Add fixed parameters to pardict
        pardict.update(self.fixedpardict)
    
        # Prepare array for planet-induced velocities
        rv_planet = np.zeros((self.nplanets, len(time)))

        for i in range(1, self.nplanets+1):
            # Skip planet if it is the one to be exluded
            if i != planet:
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


    def modelk(self, pardict, time, planet='1'):
        """
        Compute Keplerian curve.

        :param array-like param: Keplerian parameters (K, P, sqrt(e)*cos(w),
        sqrt(e)*sin(w), L0, v0, epoch)
        """

        try:
            K_ms = pardict['planet{}_k1'.format(planet)]
        except KeyError:
            K_ms = np.exp(pardict['planet{}_logk1'.format(planet)])

        try:
            P_day = pardict['planet{}_period'.format(planet)]
        except KeyError:
            P_day = np.exp(pardict['planet{}_logperiod'.format(planet)])

        ###
        # SESIN SECOS ML0
        if 'planet{}_secos'.format(planet) in pardict:
            secos = pardict['planet{}_secos'.format(planet)]
            sesin = pardict['planet{}_sesin'.format(planet)]
            ecc = secos**2 + sesin**2
            omega_rad = np.arctan2(sesin, secos)

        elif 'planet{}_ecos'.format(planet) in pardict:
            ecos = pardict['planet{}_ecos'.format(planet)]
            esin = pardict['planet{}_esin'.format(planet)]
            ecc = np.sqrt(ecos**2 + esin**2)
            omega_rad = np.arctan2(esin, ecos)

        else:
            try:
                ecc = pardict['planet{}_ecc'.format(planet)]
                omega_rad = pardict['planet{}_omega'.format(planet)]
            except KeyError:
                raise KeyError(
                    'Something is wrong with the eccentricity parametrisation')

        try:
            # Compute mean anomaly at epoch
            ml0 = pardict['planet{}_ml0'.format(planet)]
            ma0_rad = ml0 - omega_rad
        except:
            ma0_rad = pardict['planet{}_ma0'.format(planet)]

        epoch = pardict['planet{}_epoch'.format(planet)]

        # Compute mean anomaly
        ma = 2*np.pi/P_day * (time - epoch) + ma0_rad
        # Compute eccentric anomaly
        E = self.M2E(ma, ecc)
        # Compute true anomaly
        nu = self.E2v(E, ecc)

        return K_ms * (np.cos(nu + omega_rad) + ecc * np.cos(omega_rad))


    def M2E(self, M, e, ftol=1e-14, Nmax=50):
        """
        Compute eccentric anomaly from mean anomaly (and eccentricity).
        """
        E = M.copy()
        deltaE = np.array([1])
        N = 0
        while max(abs(deltaE))>ftol and N<Nmax:
            diff = M-(E-e*np.sin(E))
            deriv = 1-e*np.cos(E)
            deltaE = diff/deriv
            E += deltaE
            N += 1
        return(E)

    def E2v(self, E, e):
        """
        Compute true anomaly from eccentric anomaly (and eccentricity).
        """
        v = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
        return(v)
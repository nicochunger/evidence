"""
Module with functions concerning Priors
"""
import numpy as n
from scipy import stats, interpolate
from scipy.stats import rv_continuous

# Number of points to sample CDF and get PPF from inversion
N = 1e4
step = 1.0/N

###
# Exceptions used in this module
###
class PriorError(Exception):
    pass


###
# Home-made distributions
###
class uniform_gen(rv_continuous):
    def _argcheck(self, xmin, xmax):
        return xmin < xmax

    """
    def _pdf(self, x, xmin, xmax):
        #return stats.uniform.pdf(x, loc = xmin, scale = xmax - xmin)
        return n.where( (x >= xmin) * (x <= xmax), 1.0 / (xmax - xmin), 0.0)
    """
    def pdf(self, x, xmin, xmax):
        return n.where((x >= xmin) * (x <= xmax), 1.0 / (xmax - xmin), 0.0)

    def logpdf(self, x, xmin, xmax):
        return n.where((x >= xmin) * (x <= xmax), -n.log(xmax - xmin), -n.inf)
    
    def _cdf(self, x, xmin, xmax):
        return stats.uniform.cdf(x, loc=xmin, scale=xmax - xmin)

    # TODO Check exactly how the underscore before the method works. Is it necessary or not?
    def ppf(self, q, xmin, xmax):
        return xmin + (xmax - xmin)*q
    

class jeffreys_gen(rv_continuous):

    def _argcheck(self, xmin, xmax):
        return (xmin > 0.0) & (xmax > xmin)

    def _pdf(self, x, xmin, xmax):
        cond = n.logical_or(n.less(x, xmin), n.greater(x, xmax))
        pdf = n.where(cond, 0.0, 1.0/(x*n.log(xmax/xmin)))
        return pdf

    def _cdf(self, x, xmin, xmax):
        cdf = self._pdf(x, xmin, xmax)*x*n.log(x/xmin)
        # Consider the limits of the uniform
        cdf = n.where((x >= xmin), cdf, 0.0)
        cdf = n.where((x < xmax), cdf, 1.0)
        return cdf

    def ppf(self, q, xmin, xmax):
        return xmin * (xmax/xmin) ** q

        
class modjeff_gen(rv_continuous):
        
    def _argcheck(self, x0, xmax):
        return (xmax > x0) & (x0 > 0)

    def _pdf(self, x, x0, xmax):
        cond = n.logical_or(n.less(x, 0), n.greater(x, xmax))
        pdf = n.where(cond, 0.0, 1.0/(x0 * (1 + x/x0) * n.log(1 + xmax/x0)))
        return pdf

    def _cdf(self, x, x0, xmax):
        cdf = n.log(1 + x/x0) / n.log(1 + xmax/x0)
        cdf = n.where(x >= 0.0, cdf, 0.0)
        cdf = n.where(x < xmax, cdf, 1.0)
        return cdf

    def ppf(self, q, x0, xmax):
        return x0*((1+float(xmax)/x0) ** q) - x0

class uniformfreq_gen(rv_continuous):
        
    def _argcheck(self, xmin, xmax):
        return (xmax > xmin) & (xmin > 0)

    def _pdf(self, x, xmin, xmax):
        pdf = (xmax*xmin)/(x**2 * (xmax-xmin))
        return pdf

    def _cdf(self, x, xmin, xmax):
        cdf = xmin*xmax/(xmax-xmin) * ((1/xmin) - (1/xmin))
        cdf = n.where(x >= 0.0, cdf, 0.0)
        cdf = n.where(x < xmax, cdf, 1.0)
        return cdf

    def ppf(self, q, xmin, xmax):
        return xmin / (1 - q*(xmax-xmin)/xmax)

class binorm_gen(rv_continuous):

    def _argcheck(self, mu1, sigma1, mu2, sigma2, A):
        return (sigma1 > 0) & (sigma2 > 0) & (mu1 <= mu2) & (A >= -1.) & (A <= 1.)

    def _pdf(self, x, mu1, sigma1, mu2, sigma2, A):
        n1 = stats.norm.pdf(x, mu1, sigma1)
        n2 = stats.norm.pdf(x, mu2, sigma2)
        return 0.5*(n1*(1.-A) + n2*(1.+A))

    def _cdf(self, x, mu1, sigma1, mu2, sigma2, A):
        n1 = stats.norm.cdf(x, mu1, sigma1)
        n2 = stats.norm.cdf(x, mu2, sigma2)
        return 0.5*(n1*(1.-A) + n2*(1.+A))

    def _ppf(self, q, mu1, sigma1, mu2, sigma2, A):
        xmin = mu1 - 9.*sigma1; xmax = mu2 + 9.*sigma2
        dx = (xmax - xmin)*step
        x = n.arange(xmin, xmax + dx, dx)
        cdf = self._cdf(x, mu1, sigma1, mu2, sigma2, A)
        # Interpolate the _inverse_ CDF
        return interpolate.interp1d(cdf, x)(q)


class log10norm_gen(rv_continuous):

    def _argcheck(self, mu, sigma):
        return sigma > 0

    def _pdf(self, x, mu, sigma):
        return stats.norm.pdf(n.log10(x), mu, sigma)#/(x*n.log(10.))

    def _cdf(self, x, mu, sigma):
        return stats.norm.cdf(n.log10(x), mu, sigma)

    def _ppf(self, q, mu, sigma):
        xmin = mu - 9.*sigma; xmax = mu + 9.*sigma
        # dx = (xmax - xmin)*step
        x = n.linspace(xmin, xmax, N)
        cdf = self._cdf(10**x, mu, sigma)
        # Interpolate the _inverse_ CDF
        return 10**(interpolate.interp1d(cdf, x)(q))


# class bilog10norm_gen(rv_continuous):

#     def _argcheck(self, mu1, sigma1, mu2, sigma2, A):
#         return (sigma1 > 0) & (sigma2 > 0) & (mu1 <= mu2) & (A >= -1.) & (A <= 1.)

#     def _pdf(self, x, mu1, sigma1, mu2, sigma2, A):
#         n1 = Log10NormalPrior.pdf(x, mu1, sigma1)
#         n2 = Log10NormalPrior.pdf(x, mu2, sigma2)
#         return 0.5*(n1*(1.-A) + n2*(1.+A))

#     def _cdf(self, x, mu1, sigma1, mu2, sigma2, A):
#         xmin = mu1 - 9.*sigma1
#         xmax = mu2 + 9.*sigma2
#         xx = n.linspace(xmin, xmax, N)
#         pdf = self._pdf(10**xx, mu1, sigma1, mu2, sigma2, A)
#         cumpdf =  n.cumsum(pdf)
#         interp_cdf = interpolate.interp1d(xx, cumpdf/max(cumpdf))
#         cdf = n.zeros(len(x), float)
#         cdf[n.where(n.logical_and(x >= 10**xmin, x < 10**xmax))[0]] = interp_cdf(n.log10(x[n.where(n.logical_and(x >= 10**xmin, x < 10**xmax))[0]]))
#         cdf[n.where(x >= 10**xmax)[0]] = 1.
#         return cdf

#     def _ppf(self, q, mu1, sigma1, mu2, sigma2, A):
#         xmin = mu1 - 9.*sigma1; xmax = mu2 + 9.*sigma2
#         # dx = (xmax - xmin)*step
#         x = n.linspace(xmin, xmax, N)
#         cdf = self._cdf(10**x, mu1, sigma1, mu2, sigma2, A)
#         # Interpolate the _inverse_ CDF
#         return 10**(interpolate.interp1d(cdf, x)(q))


class asymmetricnorm_gen(rv_continuous):

    def _argcheck(self, mu, sigma1, sigma2):
        return (sigma1 > 0) & (sigma2 > 0)

    def _pdf(self, x, mu, sigma1, sigma2):
        n1 = stats.norm.pdf(x, mu, sigma1)*2.0*sigma1/(sigma1 + sigma2)
        n2 = stats.norm.pdf(x, mu, sigma2)*2.0*sigma2/(sigma1 + sigma2)
        return n.where((x <= mu), n1, n2)

    def _cdf(self, x, mu, sigma1, sigma2):
        k1 = 2.0*sigma1/(sigma1 + sigma2)
        k2 = 2.0*sigma2/(sigma1 + sigma2)
        cdf1 = stats.norm.cdf(x, mu, sigma1)*k1
        cdf2 = (stats.norm.cdf(x, mu, sigma2) - 0.5)*k2
        return n.where((x <= mu), cdf1, k1*0.5 + cdf2)

    def _ppf(self, q, mu, sigma1, sigma2):
        xmin = mu - 9*sigma1
        xmax = mu + 9*sigma2
        dx = (xmax - xmin)*step
        x = n.arange(xmin, xmax + dx, dx)
        cdf = self._cdf(x, mu, sigma1, sigma2)
        # Interpolate the _inverse_ CDF
        return interpolate.interp1d(cdf, x)(q)


class truncnormU_gen(rv_continuous):
    def _argcheck(self, mu, sigma, xmin, xmax):
        return (sigma > 0)

    def _pdf(self, x, mu, sigma, xmin, xmax):
        A1 = stats.norm.cdf(xmax, mu, sigma) - stats.norm.cdf(xmin, mu, sigma)
        n1 = stats.norm.pdf(x, mu, sigma)/A1
        return n.where((x >= xmin) & (x < xmax), n1, 0.0)

    def _cdf(self, x, mu, sigma, xmin, xmax):
        A1 = stats.norm.cdf(xmax, mu, sigma) - stats.norm.cdf(xmin, mu, sigma)
        cdf = stats.norm.cdf(x, mu, sigma) - stats.norm.cdf(xmin, mu, sigma)
        cdf = cdf/A1
        # Consider the limits of the uniform
        cdf = n.where((x >= xmin), cdf, 0.0)
        cdf = n.where((x < xmax), cdf, 1.0)
        return cdf

    def ppf(self, q, mu, sigma, xmin, xmax):
        dx = (xmax - xmin)*step
        x = n.arange(xmin, xmax + dx, dx)
        cdf = self._cdf(x, mu, sigma, xmin, xmax)
        # Interpolate the _inverse_ CDF
        return interpolate.interp1d(cdf, x)(q)

    
class truncrayleigh_gen(rv_continuous):
    def _argcheck(self, sigma, xmax):
        return (sigma > 0)

    def _pdf(self, x, sigma, xmax):
        A1 = 1 - n.exp(-xmax**2/(2*sigma**2))
        n1 = ((x/sigma**2) * n.exp(-x**2/(2*sigma**2)))/A1
        return n.where((x >= 0) & (x < xmax), n1, 0.0)

    def _cdf(self, x, sigma, xmax):
        A1 = 1 - n.exp(-xmax**2/(2*sigma**2))
        cdf = 1 - n.exp(-x**2/(2*sigma**2))
        cdf = cdf/A1
        # Consider the limits of the uniform
        cdf = n.where((x >= 0), cdf, 0.0)
        cdf = n.where((x < xmax), cdf, 1.0)
        return cdf

    def ppf(self, q, sigma, xmax):
        A = 1 - n.exp(-xmax**2/(2*sigma**2))
        ppf = n.sqrt(-2*sigma**2*n.log(1-(q*A)))
        return ppf


class truncnormJ_gen(rv_continuous):
    def _argcheck(self, mu, sigma, xmin, xmax):
        return (sigma > 0)

    def _pdf(self, x, mu, sigma, xmin, xmax):
        A1 = stats.norm.cdf(xmax, mu, sigma) - stats.norm.cdf(xmin, mu, sigma)
        n1 = stats.norm.pdf(x, mu, sigma)/A1

        ## ADDD JEFFREYS!
        return n.where((x > xmin) & (x < xmax), n1, 0.0)


class powerlaw_gen(rv_continuous):
    def _argcheck(self, alpha, xmin, xmax):
        return (xmax > xmin) and (xmin >= 0) and (xmax > 0) and (alpha != -1)

    def _pdf(self, x, alpha, xmin, xmax):
        A = (1.0 + alpha)/(xmax**(1.0 + alpha) - xmin**(1.0 + alpha))
        return n.where((x >= xmin) & (x < xmax), A*(x**alpha), 0.0)

    def _cdf(self, x, alpha, xmin, xmax):
        Aprime = 1.0/(xmax**(1.0 + alpha) - xmin**(1.0 + alpha))
        cdf = Aprime*(x**(1.0 + alpha) - xmin**(1.0 + alpha))
        cdf = n.where((x > xmin), cdf, 0.0)
        cdf = n.where((x >= xmax), 1.0, cdf)
        return cdf

    def _ppf(self, q, alpha, xmin, xmax):
        dx = (xmax - xmin)*step
        x = n.arange(xmin, xmax + dx, dx)
        cdf = self._cdf(x, alpha, xmin, xmax)
        # Interpolate the _inverse_ CDF
        return interpolate.interp1d(cdf, x)(q)


class doublepowerlaw_gen(rv_continuous):
    def _argcheck(self, alpha, beta, x0, xmin, xmax):
        return (xmax > xmin) and (xmin >= 0) and (xmax > 0) and (alpha != -1)

    def _pdf(self, x, alpha, beta, x0, xmin, xmax):
        a1 = (x0**(1.0 + alpha) - xmin**(1.0 + alpha))/(alpha + 1.0)
        a2 = (xmax**(1.0 + beta) - x0**(1.0 + beta))/(beta + 1.0)
        A = 1.0/(a1 + ((x0*1.0)**alpha/(x0*1.0)**beta)*a2)
        
        xalpha = A*x**alpha
        xbeta = ((x0*1.0)**alpha/(x0*1.0)**beta)*A*x**beta

        pdf = n.where((x < x0), xalpha, xbeta)

        return n.where((x >= xmin) & (x < xmax), pdf, 0.0)

    def _cdf(self, x, alpha, beta, x0, xmin, xmax):
        a1 = (x0**(1.0 + alpha) - xmin**(1.0 + alpha))/(alpha + 1.0)
        a2 = (xmax**(1.0 + beta) - x0**(1.0 + beta))/(beta + 1.0)
        A = 1.0/(a1 + ((x0*1.0)**alpha/(x0*1.0)**beta)*a2)

        cdfalpha = A*(x**(1.0 + alpha) - xmin**(1.0 + alpha))/(1.0 + alpha)
        cdfbeta = A*(x0**(1.0 + alpha) - xmin**(1.0 + alpha))/(1.0 + alpha) + \
            ((x0*1.0)**alpha/(x0*1.0)**beta)*A*(x**(1.0 + beta) - x0**(1.0 +beta))/(1.0 + beta)
        
        cdf = n.where((x < x0), cdfalpha, cdfbeta)

        cdf = n.where((x > xmin), cdf, 0.0)
        cdf = n.where((x >= xmax), 1.0, cdf)
        return cdf

    def _ppf(self, q, alpha, beta, x0, xmin, xmax):
        dx = (xmax - xmin)*step
        x = n.arange(xmin, xmax + dx, dx)
        cdf = self._cdf(x, alpha, beta, x0, xmin, xmax)
        # Interpolate the _inverse_ CDF
        return interpolate.interp1d(cdf, x)(q)


class sine_gen(rv_continuous):
    def _argcheck(self, xmin, xmax):
        return (xmax > xmin)

    def _pdf(self, x, xmin, xmax):
        xmin = n.where((xmin < self.a), self.a, xmin)
        xmax = n.where((xmax > self.b), self.b, xmax)
        A = 180/n.pi*(n.cos(xmin*n.pi/180.0) - n.cos(xmax*n.pi/180.0))
        pdf = n.where((x >= xmin) & (x <= xmax), n.sin(x*n.pi/180.0)/A, 0.0)
        return pdf

    def _cdf(self, x, xmin, xmax):
        xmin = n.where((xmin < self.a), self.a, xmin)
        xmax = n.where((xmax > self.b), self.b, xmax)
        A = n.cos(xmin*n.pi/180.0) - n.cos(xmax*n.pi/180.0)
        cdf = (n.cos(xmin*n.pi/180.0) - n.cos(x*n.pi/180.0))/A
        cdf = n.where((x >= xmin), cdf, 0.0)
        cdf = n.where((x <= xmax), cdf, 1.0)
        return cdf

    def _ppf(self, q, xmin, xmax):
        dx = (xmax - xmin)*step
        x = n.arange(xmin, xmax + dx, dx)
        cdf = self._cdf(x, xmin, xmax)
        # Interpolate the _inverse_ CDF
        return interpolate.interp1d(cdf, x)(q)

class alpha_gen(rv_continuous):
    """
    An alpha continuous random variable.
    %(before_notes)s
    Notes
    -----
    The probability density function for `alpha` is::
        alpha.pdf(x,a) = 1/(x**2*Phi(a)*sqrt(2*n.pi)) * exp(-1/2 * (a-1/x)**2),
    where ``Phi(alpha)`` is the normal CDF, ``x > 0``, and ``a > 0``.
    """
    def _argcheck(self, a):
        return (a > 0)
        
    def _pdf(self, x, a):
        return stats.alpha.pdf(x, a)

    def _cdf(self, x, a):
        return stats.alpha.cdf(x, a)

    def _ppf(self, q, a):
        return stats.alpha.ppf(q, a)

class beta_gen(rv_continuous):
    """
    A beta continuous random variable.
    Notes
    -----
    The probability density function for `beta` is::
        beta.pdf(x, a, b) = gamma(a+b)/(gamma(a)*gamma(b)) * x**(a-1) *
        (1-x)**(b-1),
    for ``0 < x < 1``, ``a > 0``, ``b > 0``.
    """
    def _argcheck(self, a, b):
        return (a > 0) & (b > 0)

    def _pdf(self, x, a, b):
        return stats.beta.pdf(x, a, b)

    def _cdf(self, x, a, b):
        return stats.beta.cdf(x, a, b)

    def _ppf(self, q, a, b):
        return stats.beta.ppf(q, a, b)

class gamma_gen(rv_continuous):
    """
    Notes
    -----
    The probability density function for `gamma` is::
        gamma.pdf(x, a) = lambda**a * x**(a-1) * exp(-lambda*x) / gamma(a)
    for ``x >= 0``, ``a > 0``. Here ``gamma(a)`` refers to the gamma function.
    The scale parameter is equal to ``scale = 1.0 / lambda``.
    `gamma` has a shape parameter `a` which needs to be set explicitly. For instance:
        >>> from scipy.stats import gamma
        >>> rv = gamma(3., loc = 0., scale = 2.)
    produces a frozen form of `gamma` with shape ``a = 3.``, ``loc =0.``
    and ``lambda = 1./scale = 1./2.``.
    """
    
    def _argcheck(self, alpha, beta):
        return (alpha > 0.0) & (beta > 0.0)

    def _pdf(self, x, alpha, beta):
        return stats.gamma.pdf(x, alpha, scale = 1.0/beta)

    def _cdf(self, x, alpha, beta):
        return stats.gamma.cdf(x, alpha, scale = 1.0/beta)

    def _ppf(self, q, alpha, beta):
        return stats.gamma.ppf(q, alpha, scale = 1.0/beta)

    
# Change names and define shapes parameters for the home-made distributions
Uniform = uniform_gen(name='Uniform distribution', shapes='xmin, xmax')
Jeffreys = jeffreys_gen(name='Jeffreys distribution',
                        shapes='xmin, xmax', a=0.0)
ModJeffreys = modjeff_gen(name='Modified Jeffreys distribution',
                               shapes='x0, xmax', a=0.0)
UniformFrequency = uniformfreq_gen(name='Uniform in Frequency but sampled in period',
                                    shapes='x0, xmax')
Normal = stats.norm
LogNormal = stats.lognorm
Log10Normal = log10norm_gen(name='Log10 Normal distribution',
                            shapes='mu, sigma')
Binormal = binorm_gen(name='Binormal distribution',
                      shapes='mu1, sigma1, mu2, sigma2, A')
# Log10Binormal = bilog10norm_gen(name='Log10 Binormal distribution',
#                                 shapes='mu1, sigma1, mu2, sigma2, A')
AsymmetricNormal = asymmetricnorm_gen(name='Asymmetric normal distribution',
                                      shapes='mu, sigma1, sigma2')
TruncatedUNormal = truncnormU_gen(name='Truncated normal distribution',
                                  shapes='mu, sigma, xmin, xmax')
TruncatedRayleigh = truncrayleigh_gen(name='Truncated Rayleigh distribution',
                                      shapes='sigma, xmax')
PowerLaw = powerlaw_gen(name='Power law distribution',
                        shapes='alpha, xmin, xmax')
DoublePowerLaw = doublepowerlaw_gen(name='Double Power law distribution',
                                    shapes='alpha, beta, x0, xmin, xmax')
Sine = sine_gen(name='Sine distribution', shapes='xmin, xmax', a=0.0,
                b=180.0)
Alpha = alpha_gen(name = 'Alpha distribution', shapes = 'a', a = 0.0)
Beta = beta_gen(name='Beta distribution', shapes='a, b', a=0.0, b=1.0)
# Beta = stats.beta
Gamma = gamma_gen(name='Gamma distribution', shapes='alpha, beta',
                       a=0.0)

# Construct dictionary
distdict = globals().copy()

def prior_constructor(input_dict, customprior_dict=None):
    """
    Read cofiguration file; construct dictionary with Prior instances.
    """
    priordict = {}

    # Iteration over all parameter objects
    for objkey in input_dict.keys():

        # Iteration over all parameters of a given object
        for parkey in input_dict[objkey]:

            parlist = input_dict[objkey][parkey]

            if not isinstance(parlist, list):
                continue

            # If parameter does not jump, or is marginalised skip this element
            if parlist[1] == 0:
                continue

            # Construct prior instance with information on dictionary
            priortype = parlist[2][0]
            pars = parlist[2][1:]
            try:
                #nparams = distdict[priortype][1] 
                prior = distdict[priortype](*pars)
            except KeyError:
                raise PriorError('Parameter {}_{}: Unknown type '
                                 'of prior.'.format(objkey, parkey))
            
            priordict[objkey+'_'+parkey] = prior

    return priordict


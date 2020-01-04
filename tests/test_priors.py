import sys
import pytest
import numpy as np

eps = sys.float_info.epsilon

def test_uniform():
    from evidence.priors import Uniform
    xmin = 4; xmax = 6
    prior = Uniform(xmin, xmax)
    assert prior.pdf(5) == 0.5
    assert prior.pdf(3) == 0
    assert prior.ppf(0.5) == 5
    assert prior.ppf(0.0) == 4
    assert prior.ppf(1.0) == 6
    # Integrates to 1
    step = 1e-6
    x = np.arange(xmin, xmax, step)
    y = prior.pdf(x)
    integration = np.trapz(y, x)
    print(integration)
    assert abs(integration - 1) < 1e-5

def test_jeffreys():
    from evidence.priors import Jeffreys
    xmin = 10; xmax = 100
    prior = Jeffreys(xmin, xmax)
    assert abs(prior.pdf(10) - 0.043429448190325175) < eps

    # Integrates to 1
    step = 1e-5
    x = np.arange(xmin, xmax, step)
    y = prior.pdf(x)
    integration = np.trapz(y, x)
    print(integration)
    assert abs(integration - 1) < 1e-5


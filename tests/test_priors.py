import sys
import pytest
import numpy as np

eps = sys.float_info.epsilon

def test_uniform():
    from evidence.priors import Uniform
    prior = Uniform(4,6)
    assert prior.pdf(5) == 0.5
    assert prior.pdf(3) == 0

def test_jeffreys():
    from evidence.priors import Jeffreys
    prior = Jeffreys(10, 100)
    assert abs(prior.pdf(10) - 0.043429448190325175) < eps


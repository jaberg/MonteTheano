import numpy

import theano
from theano import tensor
from theano.printing import Print

from for_theano import where

class Algo1(self):

    def __init__(self):
        pass

    def build_posterior(self, spec, observations):
        """
        Rebuild the theano graph of the spec in the form of a posterior.
        """
        raise NotImplementedError()


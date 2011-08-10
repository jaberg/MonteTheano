import numpy

import theano
from theano import tensor
from theano.printing import Print

def as_variables(*args):
    return [tensor.as_tensor_variable(a) for a in args]

class AdaptiveParzen(theano.Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, mu, low, high, minsigma):
        mu, low, high, minsigma = as_variables(mu, low, high, minsigma)
        if mu.ndim == 0:
            raise TypeError()
        if mu.ndim > 1:
            raise NotImplementedError()
        if low.ndim:
            raise TypeError(low)
        if high.ndim:
            raise TypeError(high)
        return theano.gof.Apply(self,
                [mu, low, high, minsigma],
                [mu.type(), mu.type()])

    def perform(self, node, inputs, outstorage):
        mu, low, high, minsigma = inputs
        mu_orig = mu.copy()
        mu = mu.copy()
        if len(mu) == 0:
            mu = numpy.asarray([0.5 * (low + high)])
            sigma = numpy.asarray([0.5 * (high - low)])
        elif len(mu) == 1:
            sigma = numpy.maximum(abs(mu-high), abs(mu-low))
        elif len(mu) >= 2:
            order = numpy.argsort(mu)
            mu = mu[order]
            sigma = numpy.zeros_like(mu)
            sigma[1:-1] = numpy.maximum(
                    mu[1:-1] - mu[0:-2],
                    mu[2:] - mu[1:-1])
            if len(mu)>2:
                lsigma = mu[2] - mu[0]
                usigma = mu[-1] - mu[-3]
            else:
                lsigma = mu[1] - mu[0]
                usigma = mu[-1] - mu[-2]

            sigma[0] = max(mu[0]-low, lsigma)
            sigma[-1] = max(high - mu[-1], usigma)

            # un-sort the mu and sigma
            mu[order] = mu.copy()
            sigma[order] = sigma.copy()

            print mu, sigma

            assert numpy.all(mu_orig == mu)

        outstorage[0][0] = mu
        outstorage[1][0] = numpy.maximum(sigma, minsigma)


def gauss_mixture(s_rng, mu, sigma):
    i = s_rng.random_integers(size=(),
            low=0, high=mu.shape[0]-1)
    return s_rng.normal(avg=mu[i], std=sigma[i])



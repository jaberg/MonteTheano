import numpy

import theano
from theano import tensor
from theano.printing import Print

import unittest
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


class TestAdaptiveParzen(unittest.TestCase):
    def setUp(self):
        pass

    def test_0(self):
        # mu 0
        print theano.function([],
                AdaptiveParzen()([], 0, 1, 0.01))()

    def test_1(self):
        print theano.function([],
                AdaptiveParzen()([3.0], 0, 1, 0.01))()

    def test_2(self):
        print theano.function([],
                AdaptiveParzen()([3.0, 1.0], 0, 1, .01))()

    def test_3(self):
        print theano.function([],
                AdaptiveParzen()([0.01, 0.02, 0.003, 0.7], 0, 1, .01))()


def gauss_mixture(s_rng, mu, sigma):
    i = s_rng.random_integers(size=(),
            low=0, high=mu.shape[0]-1)
    return s_rng.normal(avg=mu[i], std=sigma[i])

class Where(theano.Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x):
        if x.ndim != 1:
            raise NotImplementedError()
        return theano.gof.Apply(self,
                [x],
                [tensor.lvector()])

    def perform(self, node, inputs, outstorage):
        outstorage[0][0] = numpy.where(inputs[0])
where = Where()


def factorize(RVs):
    """
    Identify the dependency structure between a list of random variables.

    Return a dictionary such that rval[a] = set([b,c]) means there is a P(a|b,c)
    term in the graphical model. These terms are minimal in the sense that a is
    independent of the other random variables in the model given both b and c,
    and neither b nor c is irrelevant to a when conditioning on the other (of c
    and b).

    """
    raise NotImplementedError()




def sample(*args, **kwargs):
    pass

def maximize(*arg, **kwargss):
    pass

class TestGM(unittest.TestCase):

    def setUp(self):
        s_rng = tensor.shared_randomstreams.RandomStreams(23424)

        err_thresh = self.err_thresh = tensor.scalar()
        data_err = self.data_err = tensor.vector()
        data_llr = self.data_llr = tensor.vector()

        rv_err = self.rv_err = s_rng.uniform()
        rv_err_good = rv_err < err_thresh

        data_llr_good = data_llr[where(data_err < err_thresh)]
        data_llr_bad  = data_llr[where(data_err >= err_thresh)]

        # design decisions
        mu_llr_good, sigma_llr_good = AdaptiveParzen()(data_llr_good, low=-5,
                high=-1.5, minsigma=0.01)
        mu_llr_bad, sigma_llr_bad  = AdaptiveParzen()(data_llr_bad,  low=-5,
                high=-1.5, minsigma=0.01)


        rv_llr_good = gauss_mixture(s_rng, mu=mu_llr_good, sigma=sigma_llr_good)
        self.sample_llr = s_rng.normal(mean=-4, std=2, size=(5,))

        self.sample_llr_logprob = log_density(
                self.sample_llr, rv_llr_good)

        if 0:

            rv_llr_bad =  gauss_mixture(s_rng, mu=mu_llr_bad,  sigma=sigma_llr_bad)

            self.rv_llr = tensor.switch(rv_err_good, rv_llr_good, rv_llr_bad)

            self.rv_llr_interest = log_density(rv_llr_good) - log_density(rv_llr_bad)

            self.sample_llr_interest = sample(self.rv_llr_interest, size=100)

            self.lr_star = maximize(self.rv_llr_interest,
                    n_candidates = 100,
                    wrt_symbol = self.rv_llr,
                    wrt_init = self.sample_llr_interest)



    def test_rv_llr(self):
        # test that rv_llr really is a random variable

        f = theano.function(
                [self.err_thresh, self.data_err, self.data_llr],
                [self.rv_err, self.rv_llr],
                allow_input_downcast=True)

        data_err = [.0, .0, .0, .7, .7, .7]
        data_llr = [-4.5, -4, -3.5, -2, -1.5, -1.0]

        r = numpy.asarray([f(.5, data_err, data_llr) for i in xrange(100)])
        import matplotlib.pyplot as plt
        plt.scatter(r[:, 0], r[:, 1])
        plt.show()

    def test_rv_llr_interest(self):
        theano.printing.debugprint(self.rv_llr)#, 'asdf.png')


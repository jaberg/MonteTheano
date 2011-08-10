import unittest
import numpy

import theano
from theano import tensor

from .pdfreg import pdf


def SRNG(seed=2345):
    return tensor.shared_randomstreams.RandomStreams(seed)


def test_normal_simple():
    s_rng = SRNG()
    n = s_rng.normal()

    p0 = pdf(n, 0)
    p1 = pdf(n, 1)
    pn1 = pdf(n, -1)

    f = theano.function([], [p0, p1, pn1])

    pvals = f()
    targets = numpy.asarray([
                1.0 / numpy.sqrt(2*numpy.pi),
                numpy.exp(-0.5) / numpy.sqrt(2*numpy.pi),
                numpy.exp(-0.5) / numpy.sqrt(2*numpy.pi),
                ])

    assert numpy.allclose(pvals,targets), (pvals, targets)


def test_normal_w_params():
    s_rng = SRNG()
    n = s_rng.normal(avg=2, std=3)

    p0 = pdf(n, 0)
    p1 = pdf(n, 2)
    pn1 = pdf(n, -1)

    f = theano.function([], [p0, p1, pn1])

    pvals = f()
    targets = numpy.asarray([
                numpy.exp(-0.5 * ((2.0/3.0)**2)) / numpy.sqrt(2*numpy.pi*9.0),
                numpy.exp(0) / numpy.sqrt(2*numpy.pi*9),
                numpy.exp(-0.5 * ((3.0/3.0)**2)) / numpy.sqrt(2*numpy.pi*9.0),
                ])

    assert numpy.allclose(pvals,targets), (pvals, targets)


def test_normal_nonscalar():
    raise NotImplementedError()


def test_normal_w_broadcasting():
    raise NotImplementedError()


def test_uniform_simple():
    s_rng = SRNG()
    u = s_rng.uniform()

    p0 = pdf(u, 0)
    p1 = pdf(u, 1)
    p05 = pdf(u, 0.5)
    pn1 = pdf(u, -1)

    f = theano.function([], [p0, p1, p05, pn1])

    pvals = f()
    targets = numpy.asarray([1.0, 1.0, 1.0, 0.0])

    assert numpy.allclose(pvals,targets), (pvals, targets)


def test_uniform_w_params():
    s_rng = SRNG()
    u = s_rng.uniform(low=-0.999, high=9.001)

    p0 = pdf(u, 0)
    p1 = pdf(u, 2)
    p05 = pdf(u, -1.5)
    pn1 = pdf(u, 10)

    f = theano.function([], [p0, p1, p05, pn1])

    pvals = f()
    targets = numpy.asarray([.1, .1, 0, 0])
    assert numpy.allclose(pvals,targets), (pvals, targets)


def test_uniform_nonscalar():
    raise NotImplementedError()


def test_uniform_w_broadcasting():
    raise NotImplementedError()


def test_likelihood_visually():
    class A(object):pass
    self = A()

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

    if os.env.get("SHOW_PLOTS", False):

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

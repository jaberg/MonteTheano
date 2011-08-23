import unittest
import numpy
import theano
from theano import tensor
from rstreams import RandomStreams
import distributions # triggers registry
import rv
from for_theano import where

def test_dag_condition_top():
    """
    Easy test of conditioning
    """
    with RandomStreams(234) as _:
        mu = normal(10, .1)
        x = normal(mu, sigma=1)

    post_x = rv.condition([x], {mu: -7})
    theano.printing.debugprint(post_x)

    f = theano.function([], post_x)
    r = [f() for i in range(10)]
    assert numpy.allclose(numpy.mean(r), -7.4722755432)


def test_dag_condition_bottom():
    """
    Test test of conditioning an upper node on a lower one
    """
    with RandomStreams(234) as _:
        mu = normal(10, .1)
        x = normal(mu, sigma=1)

    post_mu = rv.condition([mu], {x: -7})
    theano.printing.debugprint(post_mu)

    f = theano.function([], post_mu)
    f()


def test_normal_simple():
    s_rng = RandomStreams(23)
    n = s_rng.normal()

    p0 = rv.lpdf(n, 0)
    p1 = rv.lpdf(n, 1)
    pn1 = rv.lpdf(n, -1)

    f = theano.function([], [p0, p1, pn1])

    pvals = f()
    targets = numpy.asarray([
                numpy.log(1.0 / numpy.sqrt(2*numpy.pi)),
                numpy.log(numpy.exp(-0.5) / numpy.sqrt(2*numpy.pi)),
                numpy.log(numpy.exp(-0.5) / numpy.sqrt(2*numpy.pi)),
                ])

    assert numpy.allclose(pvals,targets), (pvals, targets)


def test_normal_w_params():
    s_rng = RandomStreams(23)
    n = s_rng.normal(mu=2, sigma=3)

    p0 = rv.lpdf(n, 0)
    p1 = rv.lpdf(n, 2)
    pn1 = rv.lpdf(n, -1)

    f = theano.function([], [p0, p1, pn1])

    pvals = f()
    targets = numpy.asarray([
                numpy.log(numpy.exp(-0.5 * ((2.0/3.0)**2)) /
                    numpy.sqrt(2*numpy.pi*9.0)),
                numpy.log(numpy.exp(0) / numpy.sqrt(2*numpy.pi*9)),
                numpy.log(numpy.exp(-0.5 * ((3.0/3.0)**2)) /
                    numpy.sqrt(2*numpy.pi*9.0)),
                ])

    assert numpy.allclose(pvals,targets), (pvals, targets)


def test_normal_nonscalar():
    s_rng = RandomStreams(234)
    n = s_rng.normal()

    data = numpy.asarray([1, 2, 3, 4, 5])
    p_data = rv.lpdf(n, data)

    f = theano.function([], [p_data])

    pvals = f()
    targets = numpy.log(numpy.exp(-0.5 * (data**2)) / numpy.sqrt(2*numpy.pi))

    assert numpy.allclose(pvals,targets), (pvals, targets)


def test_normal_w_broadcasting():
    raise NotImplementedError()


def test_uniform_simple():
    s_rng = RandomStreams(234)
    u = s_rng.uniform()

    p0 = rv.lpdf(u, 0)
    p1 = rv.lpdf(u, 1)
    p05 = rv.lpdf(u, 0.5)
    pn1 = rv.lpdf(u, -1)

    f = theano.function([], [p0, p1, p05, pn1])

    pvals = f()
    targets = numpy.log(numpy.asarray([1.0, 1.0, 1.0, 0.0]))

    assert numpy.allclose(pvals,targets), (pvals, targets)


def test_uniform_w_params():
    s_rng = RandomStreams(234)
    u = s_rng.uniform(low=-0.999, high=9.001)

    p0 = rv.lpdf(u, 0)
    p1 = rv.lpdf(u, 2)
    p05 = rv.lpdf(u, -1.5)
    pn1 = rv.lpdf(u, 10)

    f = theano.function([], [p0, p1, p05, pn1])

    pvals = f()
    targets = numpy.log(numpy.asarray([.1, .1, 0, 0]))
    assert numpy.allclose(pvals,targets), (pvals, targets)


def test_uniform_nonscalar():
    raise NotImplementedError()


def test_uniform_w_broadcasting():
    raise NotImplementedError()


if 0:
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
    self.sample_llr = s_rng.normal(avg=-4, std=2, size=(5,))

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

test_normal_nonscalar()

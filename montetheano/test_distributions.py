import unittest
import numpy

import theano
from theano import tensor
from distributions import sample, likelihood

class TestHierarchicalNormal(unittest.TestCase):
    def setUp(self):
        s_rng = self.s_rng = tensor.shared_randomstreams.RandomStreams(23424)
        a = 0.0
        b = 1.0
        c = 1.5
        d = 2.0

        self.M = s_rng.normal((), a, b)
        self.V = abs(s_rng.normal((), c, d)) + .1
        self.X = s_rng.normal((4,), self.M, self.V)

        X_data = tensor.as_tensor_variable([[1, 2, 3, 2.4]])

    def test_sample_gets_all_rvs(self):
        outs, dct = sample(self.s_rng, [self.X], ())
        assert outs == [self.X]
        assert len(dct) == 3

    def test_sample_can_be_generated(self):
        outs, dct = sample(self.s_rng, [self.X], ())
        f = theano.function([], [dct[self.X], dct[self.M],
            dct[self.V.owner.inputs[0]]])
        x0, m0, v0 = f()
        x1, m1, v1 = f()
        assert not numpy.any(x0 == x1)
        assert x0.shape == (4,)
        assert m0.shape == ()
        assert v1.shape == ()
        print x0, m0, v0

    def test_likelihood(self):


    def test_mh_sample(self):
        posterior = mh_sample(s_rng,
                observations={self.X: X_data},
                size=(100,),
                lag=50,
                burnin=200)

        self.M_samples = posterior[M]
        self.V_samples = posterior[V]


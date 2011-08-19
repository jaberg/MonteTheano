import unittest
import numpy

import theano
from theano import tensor

from rstreams import RandomStreams
import distributions
from sample import rejection_sample, mh_sample, hybridmc_sample
from rv import full_log_likelihood

import pylab

class TestBasicBinomial(unittest.TestCase):
    def setUp(self):
        s_rng = self.s_rng = RandomStreams(23424)

        p = 0.5
        
        self.A = s_rng.binomial(1, p)
        self.B = s_rng.binomial(1, p)
        self.C = s_rng.binomial(1, p)
        
        self.D = self.A+self.B+self.C
        
        self.condition = tensor.ge(self.D, 2)
        
    def test_rejection_sampler(self):
        sample, updates = rejection_sample([self.A, self.B, self.C], self.condition)
        
        # create a runnable function
        sampler = theano.function(inputs=[], outputs = sample, updates = updates)

        # generate some data
        data = []
        for i in range(100):
            data.append(sampler())

        # plot histogram
        pylab.hist(numpy.asarray(data))
        pylab.show()

    def test_rejection_sampler_no_cond(self):
        sample, updates = rejection_sample([self.A, self.B, self.C])
        
        # create a runnable function
        sampler = theano.function(inputs=[], outputs = sample, updates = updates)

        # generate some data
        data = []
        for i in range(100):
            data.append(sampler())

        # plot histogram
        pylab.hist(numpy.asarray(data))
        pylab.show()

# first example: http://projects.csail.mit.edu/church/wiki/Learning_as_Conditional_Inference
class TestCoin(unittest.TestCase):
    def setUp(self):
        s_rng = self.s_rng = RandomStreams(23424)

        self.fair_prior = 0.999
        self.fair_coin = s_rng.binomial(1, self.fair_prior)
        
        make_coin = lambda x: s_rng.binomial((4,), 1, x)    
        self.coin = make_coin(tensor.switch(self.fair_coin > 0.5, 0.5, 0.95))

        self.data = tensor.as_tensor_variable([[1, 1, 1, 1]])
        
    def test_tt(self):
        sample, updates = rejection_sample([self.fair_coin,], tensor.eq(tensor.sum(tensor.eq(self.coin, self.data)), 5))
        sampler = theano.function([], sample, updates=updates)
        
        # TODO: this is super-slow, how can bher do this fast?
        for i in range(100):
            print sampler()

class TestCoin2(): #unittest.TestCase):
    def setUp(self):
        s_rng = self.s_rng = RandomStreams(23424)

        self.repetitions = 100        
        self.coin_weight = s_rng.uniform(low=0, high=1)
        self.coin = s_rng.binomial((self.repetitions,), 1, self.coin_weight)
        
    def test_tt(self):
        true_sampler = theano.function([self.coin_weight], self.coin)

        sample, ll, updates = mh_sample(self.s_rng, [self.coin_weight])
        sampler = theano.function([self.coin], sample, updates=updates)

        for i in range(100):
            print sampler(true_sampler(0.9))
        
class TestGMM(unittest.TestCase):
    def setUp(self):
        s_rng = self.s_rng = RandomStreams(23424)

        self.p = tensor.scalar()
        self.m1 = tensor.scalar() 
        self.m2 = tensor.scalar() 
        self.v = tensor.scalar() 
        
        self.C = s_rng.binomial(1, p)
        self.m = tensor.switch(self.C, self.m1, self.m2)
        self.D = s_rng.normal(self.m, self.v)        
    
        self.D_data = tensor.as_tensor_variable([1, 1.2, 3, 3.4])
        
    def test_tt(self):
        RVs = dict([(self.D, self.D_data)])
        lik = full_log_likelihood(RVs)
        
        lf = theano.function([self.m1, self.m2, self.C], lik)
        
        print lf(1,3,0)
        print lf(1,3,1)
        
        # EM:
        #     E-step:
        #         C = expectation p(C | data, params)
        #     M-step:
        #         params = argmax p(params | C, data)
        # 
        # MCMC (Gibbs):
        #     p(params | data, C)
        #     p(C | data, params)
        
        
class TestHierarchicalNormal(): #unittest.TestCase):
    def setUp(self):
        s_rng = self.s_rng = RandomStreams(23424)
        a = 0.0
        b = 1.0
        c = 1.5
        d = 2.0

        self.M = s_rng.normal(a, b)
        self.V = s_rng.normal(c, d)
        self.V_ = abs(self.V) + .1
        self.X = s_rng.normal((4,), self.M, self.V_)

        self.X_data = tensor.as_tensor_variable([1, 2, 3, 2.4])

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
        outs, obs = sample(self.s_rng, [self.X], ())

        lik = likelihood(obs)

        f = theano.function([], lik)

        print f()

    def test_mh_sample(self):
        sample, ll, updates = mh_sample(self.s_rng, [self.M, self.V], observations={self.X: self.X_data}, lag = 100)
        sampler = theano.function([], sample, updates=updates)
        
        data = []
        for i in range(100):
            print i
            data.append(sampler())
        
        pylab.subplot(211)
        pylab.hist(numpy.asarray(data)[:,0])
        pylab.subplot(212)
        pylab.hist(numpy.asarray(data)[:,1])
        pylab.show()
        
class TestBayesianLogisticRegression(): #unittest.TestCase):
    def setUp(self):
        s_rng = self.s_rng = RandomStreams(3424)

        self.W1 = s_rng.normal(0, 4)
        self.W2 = s_rng.normal(0, 4)
        
        self.x = tensor.matrix('x')
        # self.y = tensor.nnet.sigmoid(tensor.dot(self.x, self.W) + self.b)
        self.y = tensor.nnet.sigmoid(self.x[:,0]*self.W1 + self.x[:,1]*self.W2)

        self.t = s_rng.binomial(shape=(4,), p=self.y)
        
        self.X_data = numpy.asarray([[-1.5, -0.4, 1.3, 2.2],[-1.1, -2.2, 1.3, 0]], dtype=theano.config.floatX).T 
        self.Y_data = numpy.asarray([1., 1., 0., 0.], dtype=theano.config.floatX)

    def test_likelihood(self):            
        RVs = dict([(self.t, self.Y_data)])                
        lik = full_log_likelihood(RVs)
        
        givens = dict([(self.x, self.X_data)])
        lik_func = theano.function([self.W1, self.W2], lik, givens=givens, allow_input_downcast=True)

        delta = .1
        x = numpy.arange(-10.0, 10.0, delta)
        y = numpy.arange(-10.0, 10.0, delta)
        X, Y = numpy.meshgrid(x, y)

        response = []
        for x, y in zip(X.flatten(), Y.flatten()):
            response.append(lik_func(x, y))

        pylab.figure(1)
        pylab.contour(X, Y, numpy.exp(numpy.asarray(response)).reshape(X.shape), 20)            
        pylab.draw()

        # sample, ll, updates = mh_sample(self.s_rng, [self.W1, self.W2], observations={self.t: self.Y_data})
        sample, ll, updates = hybridmc_sample(self.s_rng, [self.W1, self.W2], observations={self.t: self.Y_data})

        sampler = theano.function([], sample + [ll] , updates=updates, givens={self.x: self.X_data}, allow_input_downcast=True)
        out = theano.function([self.W1, self.W2, self.x], self.y, allow_input_downcast=True)
        
        delta = 0.1
        x = numpy.arange(-3, 3, delta)
        y = numpy.arange(-3, 3, delta)
        X, Y = numpy.meshgrid(x, y)

        b = numpy.zeros(X.shape)
        for i in range(500):
            W1, W2, ll = sampler()            

            if i % 10 == 0:
                pylab.figure(1)            
                pylab.plot(W1, W2, 'x')
                pylab.draw()

                response = out(W1, W2, numpy.vstack((X.flatten(), Y.flatten())).T)
                response = response.reshape(X.shape)
                b += response

                pylab.figure(2)
                pylab.contour(X, Y, response)            
                pylab.plot(self.X_data[:2,1], self.X_data[:2,0], 'kx')
                pylab.plot(self.X_data[2:,1], self.X_data[2:,0], 'bo')
                pylab.draw()
                pylab.clf()

        pylab.figure(1)
        pylab.clf()
        pylab.contour(X, Y, b)            
        pylab.plot(self.X_data[:2,0], self.X_data[:2,1], 'kx')
        pylab.plot(self.X_data[2:,0], self.X_data[2:,1], 'bo')
        pylab.show()

class Fitting1D(unittest.TestCase):
    def setUp(self):
        self.obs = tensor.as_tensor_variable(
                numpy.asarray([0.0, 1.01, 0.7, 0.65, 0.3]))
        self.rstream = RandomStreams(234)
        self.n = self.rstream.normal()
        self.u = self.rstream.uniform()

    def test_normal_ml(self):
        up = self.rstream.ml(self.n, self.obs)
        p = self.rstream.params(self.n)
        f = theano.function([], [up[p[0]], up[p[1]]])
        m,v = f()
        assert numpy.allclose([m,v], [.532, 0.34856276335])

    def test_uniform_ml(self):
        up = self.rstream.ml(self.u, self.obs)
        p = self.rstream.params(self.u)
        f = theano.function([], [up[p[0]], up[p[1]]])
        l,h = f()
        assert numpy.allclose([l,h], [0.0, 1.01])

t = TestBayesianLogisticRegression()
t.setUp()
t.test_likelihood()

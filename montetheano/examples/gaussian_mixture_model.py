import numpy, pylab
import theano
from theano import tensor
from rstreams import RandomStreams
import distributions
from sample import mh2_sample
from rv import full_log_likelihood

s_rng = RandomStreams(3424)

p = s_rng.dirichlet(numpy.asarray([1, 1]))[0]
m1 = s_rng.uniform(low=-5, high=5)
m2 = s_rng.uniform(low=-5, high=5)
v = s_rng.uniform(low=0, high=1)

C = s_rng.binomial(1, p, draw_shape=(4,))
m = tensor.switch(C, m1, m2)
D = s_rng.normal(m, v, draw_shape=(4,))        

D_data = numpy.asarray([1, 1.2, 3, 3.4], dtype=theano.config.floatX)

givens = dict([(D, D_data)])
sampler = mh2_sample(s_rng, [p, m1, m2, v], givens)            

samples = sampler(200, 1000, 100)
print samples[0].mean(), samples[1].mean(), samples[2].mean(), samples[3].mean()

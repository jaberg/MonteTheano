import numpy
import theano
from theano import tensor
from rstreams import RandomStreams
import distributions
from sample import hybridmc_sample
from rv import full_log_likelihood

from max_lik import likelihood_gradient 

s_rng = RandomStreams(3424)

# Weight prior:
w = s_rng.normal(0, 2, draw_shape=(3,))

# Linear model:
x = tensor.matrix('x')
y = tensor.nnet.sigmoid(tensor.dot(x, w))

# Bernouilli observation model:
t = s_rng.binomial(p=y, draw_shape=(4,))

# Some data:
X_data = numpy.asarray([[-1.5, -0.4, 1.3, 2.2], [-1.1, -2.2, 1.3, 0], [1., 1., 1., 1.]], dtype=theano.config.floatX).T 
Y_data = numpy.asarray([1., 1., 0., 0.], dtype=theano.config.floatX)

# Compute gradient updates:
observations = dict([(t, Y_data)])
params, updates, log_likelihood = likelihood_gradient(observations)

# Compile training function and assign input data as givens:
givens = dict([(x, X_data)])
train = theano.function([], [log_likelihood], givens=givens, updates=updates)

# Run 100 epochs of training:
for i in range(100):
    print "epoch", i, ", log likelihood:", train()[0]

    
# Generate testing function:    
givens = dict([(x, X_data)]) 
givens.update(params)
test = theano.function([], [y], givens=givens)

print test(), Y_data
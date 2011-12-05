import numpy, pylab
import theano
from theano import tensor
from rstreams import RandomStreams
import distributions
from sample import mh2_sample
from rv import full_log_likelihood
from for_theano import evaluate

s_rng = RandomStreams(3424)

def poly_expansion(x, order):
	x = x.T
	result, updates = theano.scan(fn=lambda prior_result, x: prior_result * x,
			outputs_info=tensor.ones_like(x),
			non_sequences=x,
			n_steps=order)
			
	return tensor.concatenate([tensor.ones([x.shape[1],1]), tensor.reshape(result.T, (x.shape[1], x.shape[0]*order))], axis=1)

# Define priors to be inverse gamma distributions
alpha = 1/s_rng.gamma(1., 2.)
beta = 1/s_rng.gamma(1., .1)

# Order of the model
# TODO: this currently has to be fixed, would be nice if this could also be a RV!
m = 7 #s_rng.random_integers(1, 10)
w = s_rng.normal(0, beta, draw_shape=(m+1,))

# Input variable used for training
x = tensor.matrix('x')
# Input variable used for testing
xn = tensor.matrix('xn')

# Actual linear model
y = lambda x_in: tensor.dot(poly_expansion(x_in, m), w)

# Observation model
t = s_rng.normal(y(x), alpha, draw_shape=(10,))

# Generate some noisy training data (sine + noise)
X_data = numpy.arange(-1,1,0.3)
Y_data = numpy.sin(numpy.pi*X_data) + 0.1*numpy.random.randn(*X_data.shape)
X_data.shape = (X_data.shape[0],1)

X_new = numpy.arange(-1,1,0.05)
X_new.shape = (X_new.shape[0],1)

pylab.plot(X_data, Y_data, 'x', markersize=10)

# Generate samples from the model
sampler = mh2_sample(s_rng, [y(xn)], observations={t: Y_data}, givens={x: X_data, xn: X_new})            
samples = sampler(50, 1000, 200)
pylab.errorbar(X_new, numpy.mean(samples[0].T, axis=1), numpy.std(samples[0].T, axis=1))
pylab.show()
pylab.plot(X_new, samples[0].T)
pylab.show()

import numpy, pylab
import theano
from theano import tensor
from rstreams import RandomStreams
import distributions
from sample import hybridmc_sample
from rv import full_log_likelihood

s_rng = RandomStreams(3424)

w = s_rng.normal(0, 4, draw_shape=(2,))

x = tensor.matrix('x')
y = tensor.nnet.sigmoid(tensor.dot(x, w))

t = s_rng.binomial(p=y, draw_shape=(4,))

X_data = numpy.asarray([[-1.5, -0.4, 1.3, 2.2],[-1.1, -2.2, 1.3, 0]], dtype=theano.config.floatX).T 
Y_data = numpy.asarray([1., 1., 0., 0.], dtype=theano.config.floatX)

RVs = dict([(t, Y_data)])                
lik = full_log_likelihood(RVs)

givens = dict([(x, X_data)])
lik_func = theano.function([w], lik, givens=givens, allow_input_downcast=True)

delta = .1
x_range = numpy.arange(-10.0, 10.0, delta)
y_range = numpy.arange(-10.0, 10.0, delta)
X, Y = numpy.meshgrid(x_range, y_range)

response = []
for xl, yl in zip(X.flatten(), Y.flatten()):
    response.append(lik_func([xl, yl]))

pylab.figure(1)
pylab.contour(X, Y, numpy.exp(numpy.asarray(response)).reshape(X.shape), 20)            
pylab.draw()

sample, ll, updates = hybridmc_sample(s_rng, [w], observations={t: Y_data})

sampler = theano.function([], sample + [ll] , updates=updates, givens={x: X_data}, allow_input_downcast=True)
out = theano.function([w, x], y, allow_input_downcast=True)

delta = 0.1
x_range = numpy.arange(-3, 3, delta)
y_range = numpy.arange(-3, 3, delta)
X, Y = numpy.meshgrid(x_range, y_range)

b = numpy.zeros(X.shape)
for i in range(1000):
    w, ll = sampler()            

    if i % 50 == 0:
        pylab.figure(1)            
        pylab.plot(w[0], w[1], 'x')
        pylab.draw()

        response = out(w, numpy.vstack((X.flatten(), Y.flatten())).T)
        response = response.reshape(X.shape)
        b += response

        pylab.figure(2)
        pylab.contour(X, Y, response)            
        pylab.plot(X_data[:2,1], X_data[:2,0], 'kx')
        pylab.plot(X_data[2:,1], X_data[2:,0], 'bo')
        pylab.draw()
        pylab.clf()

pylab.figure(1)
pylab.clf()
pylab.contour(X, Y, b)            
pylab.plot(X_data[:2,0], X_data[:2,1], 'kx')
pylab.plot(X_data[2:,0], X_data[2:,1], 'bo')
pylab.show()

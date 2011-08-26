"""
Math for various distributions.

"""
import __builtin__
import copy
import numpy
import theano
import scipy
import scipy.special
from theano import tensor
from for_theano import elemwise_cond, ancestors, infer_shape, evaluate
from rstreams import rng_register

# TODOs:
# - Additional distributions of interest:
#   - Wishart
#   - Dirichlet process / CRP

# -------
# Uniform
# -------


@rng_register
def uniform_sampler(rstream, low=0.0, high=1.0, ndim=None, draw_shape=None, dtype=theano.config.floatX):
    low = tensor.as_tensor_variable(low)
    high = tensor.as_tensor_variable(high)
    rstate = rstream.new_shared_rstate()

    # James: why is this required? fails in draw_shape is not provided
    # if isinstance(draw_shape, (list, tuple)):
    #     draw_shape = tensor.stack(*draw_shape)

    new_rstate, out = tensor.raw_random.uniform(rstate, draw_shape, low, high, ndim, dtype)
    rstream.add_default_update(out, rstate, new_rstate)
    return out


@rng_register
def uniform_lpdf(node, sample, kw):
    rstate, shape, low, high = node.inputs
    rval = elemwise_cond(
        numpy.array(float('-inf')), sample < low,
        -tensor.log(high-low), sample <= high,
        numpy.array(float('-inf')))
    return rval

@rng_register
def uniform_ml(node, sample, weights):
    rstate, shape, low, high = node.inputs
    return Updates({
        low: sample.min(),
        high: sample.max()})

@rng_register
def uniform_params(node):
    rstate, shape, low, high = node.inputs
    return [low, high]

# ------
# Normal
# ------


@rng_register
def normal_sampler(rstream, mu=0.0, sigma=1.0, draw_shape=None, ndim=None, dtype=None):
    mu = tensor.as_tensor_variable(mu)
    sigma = tensor.as_tensor_variable(sigma)
    rstate = rstream.new_shared_rstate()

    new_rstate, out = tensor.raw_random.normal(rstate, draw_shape, mu, sigma, dtype=dtype)
    rstream.add_default_update(out, rstate, new_rstate)
    return out


@rng_register
def normal_lpdf(node, sample, kw):
    # make sure that the division is done at least with float32 precision
    one = tensor.as_tensor_variable(numpy.asarray(1, dtype='float32'))
    rstate, shape, mu, sigma = node.inputs
    Z = tensor.sqrt(2 * numpy.pi * sigma**2)
    E = 0.5 * ((mu - sample)/(one*sigma))**2
    return - E - tensor.log(Z)

@rng_register
def normal_ml(node, sample, weights):
    rstate, shape, mu, sigma = node.inputs
    eps = 1e-8
    if weights is None:
        new_mu = tensor.mean(sample)
        new_sigma = tensor.std(sample)

    else:
        denom = tensor.maximum(tensor.sum(weights), eps)
        new_mu = tensor.sum(sample*weights) / denom
        new_sigma = tensor.sqrt(
                tensor.sum(weights * (sample - new_mu)**2)
                / denom)
    return Updates({
        mu: new_mu,
        sigma: new_sigma})

@rng_register
def normal_params(node):
    rstate, shape, mu, sigma = node.inputs
    return [mu, sigma]

@rng_register
def normal_proposal(rstream, node, sample, kw):
    return rstream.normal(sample, 0.1, draw_shape = infer_shape(node.outputs[1]))


# ---------
# Binomial
# ---------

@rng_register
def binomial_sampler(rstream, n=1, p=0.5, ndim=0, draw_shape=None, dtype=theano.config.floatX):
    if not isinstance(n, theano.Variable):
        n = tensor.shared(numpy.asarray(n, dtype=int))
    if not isinstance(p, theano.Variable):
        p = tensor.shared(numpy.asarray(p, dtype=theano.config.floatX))
    rstate = rstream.new_shared_rstate()

    new_rstate, out = tensor.raw_random.binomial(rstate, draw_shape, n, p, dtype=dtype)
    rstream.add_default_update(out, rstate, new_rstate)
    return out

@rng_register
def binomial_lpdf(node, x, kw):
    random_state, size, n, p = node.inputs

    # for the n > 1 the "choose" operation is required
    # TODO assert n == 1
    
    return tensor.switch(tensor.eq(x, 1.), tensor.log(p), tensor.log(1. - p))

@rng_register
def binomial_params(node):
    rstate, shape, n, p = node.inputs
    return [n, p]

# ---------
# LogNormal
# ---------


@rng_register
def lognormal_sampler(rstream, mu=0.0, sigma=1.0, draw_shape=None, ndim=None, dtype=theano.config.floatX):
    """
    Sample from a normal distribution centered on avg with
    the specified standard deviation (std).

    If the size argument is ambiguous on the number of dimensions, ndim
    may be a plain integer to supplement the missing information.

    If size is None, the output shape will be determined by the shapes
    of avg and std.

    If dtype is not specified, it will be inferred from the dtype of
    avg and std, but will be at least as precise as floatX.
    """
    mu = tensor.as_tensor_variable(mu)
    sigma = tensor.as_tensor_variable(sigma)

    if dtype == None:
        dtype = tensor.scal.upcast(
                theano.config.floatX, mu.dtype, sigma.dtype)
    rstate = rstream.new_shared_rstate()

    ndim, draw_shape, bcast = tensor.raw_random._infer_ndim_bcast(
            ndim, draw_shape, mu, sigma)
    op = tensor.raw_random.RandomFunction('lognormal',
            tensor.TensorType(dtype=dtype, broadcastable=bcast))
    new_rstate, out = op(rstate, draw_shape, mu, sigma)
    rstream.add_default_update(out, rstate, new_rstate)
    return out

def lognormal_cdf(x, mu, sigma):
    # wikipedia claims cdf is
    # .5 + .5 erf( log(x) - mu / sqrt(2 sigma^2))
    return .5 + .5 * tensor.erf(
            (tensor.log(x) - mu)
            / tensor.sqrt(2 * sigma**2))

@rng_register
def lognormal_lpdf(node, x, kw):
    r, shape, mu, sigma = node.inputs
    if 'float' in node.outputs[1].dtype:
        # formula copied from wikipedia
        # http://en.wikipedia.org/wiki/Log-normal_distribution
        Z = sigma * x * numpy.sqrt(2 * numpy.pi)
        E = 0.5 * ((tensor.log(x) - mu) / sigma)**2
        return -E - tensor.log(Z)
    elif 'int' in node.outputs[1].dtype:
        # casting rounds down to nearest non-negative integer.
        # so lpdf is log of integral from x to x+1 of P(x)
        #
        # TODO: subtracting these two numbers that are really close together and
        # then taking the log of that difference sounds numerically terrible.
        return tensor.log(
                lognormal_cdf(x+1, mu, sigma)
                - lognormal_cdf(x, mu, sigma))
    else:
        raise NotImplementedError()

# -----------
# Categorical
# -----------


class Categorical(theano.Op):
    dist_name = 'categorical'
    def __init__(self, destructive, otype):
        self.destructive = destructive
        self.otype = otype
        if destructive:
            self.destroy_map = {0:[0]}
        else:
            self.destroy_map = {}

    def __eq__(self, other):
        return (type(self) == type(other)
                and self.destructive == other.destructive
                and self.otype == other.otype)

    def __hash__(self):
        return hash((type(self), self.destructive, self.otype))

    def make_node(self, s_rstate, p, draw_shape):
        p = tensor.as_tensor_variable(p)
        draw_shape = tensor.as_tensor_variable(draw_shape)
        return theano.gof.Apply(self,
                [s_rstate, p, draw_shape],
                [s_rstate.type(), self.otype()])

    def perform(self, node, inputs, outstor):
        rng, p, shp = inputs
        if not self.destructive:
            rng = copy.deepcopy(rng)
        n_draws = numpy.prod(shp)
        rval = [numpy.where(rng.multinomial(pvals=p, n=1))[0][0]
                for i in xrange(n_draws)]
        rval = numpy.asarray(rval, dtype=self.otype.dtype)
        rval.shape = shp
        outstor[0][0] = rng
        outstor[1][0] = self.otype.filter(rval, allow_downcast=True)


@rng_register
def categorical_sampler(rstream, p, draw_shape, dtype='int32'):
    if not isinstance(p, theano.Variable):
        p = tensor.shared(numpy.asarray(p, dtype=theano.config.floatX))
    if p.ndim != 1:
        raise NotImplementedError()
    if draw_shape.ndim != 1:
        raise TypeError()
    op = Categorical(False,
            tensor.TensorType(
                broadcastable=(False,)* tensor.get_vector_length(draw_shape),
                dtype=dtype))
    rstate = rstream.new_shared_rstate()
    new_rstate, out = op(rstate, p, draw_shape)
    rstream.add_default_update(out, rstate, new_rstate)
    return out


@rng_register
def categorical_lpdf(node, sample, kw):
    """
    Return a random integer from 0 .. N-1 inclusive according to the
    probabilities p[0] .. P[N-1].

    This is formally equivalent to numpy.where(multinomial(n=1, p))
    """
    # WARNING: I think the p[-1] is not used, but assumed to be p[:-1].sum()
    s_rstate, p, draw_shape = node.inputs
    return p[sample]


# ---------
# LogGamma helper Op
# ---------

# class PolyGamma(theano.Op):
#     def __eq__(self, other):
#         return type(self) == type(other)
# 
#     def __hash__(self):
#         return hash(type(self))
# 
#     def make_node(self, x):
#         x_ = tensor.as_tensor_variable(x).astype(theano.config.floatX)    
#         return theano.Apply(self,
#             inputs=[x_],
#             outputs=[x_.type()])
# 
#     def perform(self, node, inputs, output_storage):      
#         x, = inputs
#         output_storage[0][0] = numpy.asarray(scipy.special.polygamma(0, x), dtype=node.outputs[0].dtype)
#         
# polyGamma = PolyGamma()

class LogGamma(theano.Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x):
        x_ = tensor.as_tensor_variable(x).astype(theano.config.floatX)    
        return theano.Apply(self,
            inputs=[x_],
            outputs=[x_.type()])

    def perform(self, node, inputs, output_storage):      
        x, = inputs
        output_storage[0][0] = numpy.asarray(scipy.special.gammaln(x), dtype=node.outputs[0].dtype)

    # TODO: is this correct ?
    # def grad(self, inp, grads):
    #     s, = inp
    #     dt, = grads        
    #     return [polyGamma(s)*dt]

logGamma = LogGamma()

# ---------
# Dirichlet
# ---------

@rng_register
def dirichlet_sampler(rstream, alpha, draw_shape=None, ndim=None, dtype=theano.config.floatX):
    tmp = alpha.T[0].T

    alpha = tensor.as_tensor_variable(alpha).astype(theano.config.floatX)
    if dtype == None:
        dtype = tensor.scal.upcast(theano.config.floatX, alpha.dtype)
        
    ndim, draw_shape, bcast = tensor.raw_random._infer_ndim_bcast(ndim, draw_shape, tmp)
    bcast = bcast+(alpha.type.broadcastable[-1],)
    
    op = tensor.raw_random.RandomFunction('dirichlet',
            tensor.TensorType(dtype=dtype, broadcastable=bcast), ndim_added=1)
        
    rstate = rstream.new_shared_rstate()
    new_rstate, out = op(rstate, draw_shape, alpha)
    rstream.add_default_update(out, rstate, new_rstate)
    return out

def logBeta(alpha):
    return tensor.sum(logGamma(alpha)) - logGamma(tensor.sum(alpha))
    
@rng_register
def dirichlet_lpdf(node, sample, kw):
    r, shape, alpha = node.inputs

    # assert sum(sample) == 1
    
    stable = tensor.eq(0, (tensor.sum(alpha <= 0.) + tensor.sum(sample <= 0.)))    
    ll = -logBeta(alpha) + tensor.sum(tensor.log(sample)*(alpha-1.), axis=0)    
    return tensor.switch(stable, ll, tensor.as_tensor_variable(float('-inf')))
            
@rng_register
def dirichlet_proposal(rstream, node, sample, kw):
    return node.outputs[1]
    
# ---------
# Gamma
# ---------

@rng_register
def gamma_sampler(rstream, k, theta, draw_shape=None, ndim=None, dtype=theano.config.floatX):
    k = tensor.as_tensor_variable(k)
    theta = tensor.as_tensor_variable(theta)
    if dtype == None:
        dtype = tensor.scal.upcast(theano.config.floatX, k.dtype, theta.dtype)    
        
    ndim, draw_shape, bcast = tensor.raw_random._infer_ndim_bcast(ndim, draw_shape, k, theta)
    op = tensor.raw_random.RandomFunction('gamma',
            tensor.TensorType(dtype=dtype, broadcastable=bcast))
            
    rstate = rstream.new_shared_rstate()
    new_rstate, out = op(rstate, draw_shape, k, theta)
    rstream.add_default_update(out, rstate, new_rstate)
    return out

@rng_register
def gamma_lpdf(node, x, kw):
    r, shape, k, theta = node.inputs

    return tensor.log(x)*(k-1.) - x/theta - tensor.log(theta)*k - logGamma(k)
    
@rng_register
def gamma_proposal(rstream, node, sample, kw):
    # return rstream.lognormal(tensor.log(sample), 1)
    return node.outputs[1]

# ---------
# Multinomial
# ---------

@rng_register
def multinomial_sampler(rstream, n=1, p=[0.5, 0.5], draw_shape=None, ndim=None, dtype=theano.config.floatX):
    if not isinstance(n, theano.Variable):
        n = tensor.shared(numpy.asarray(n, dtype=int))
    if not isinstance(p, theano.Variable):
        p = tensor.shared(numpy.asarray(p, dtype=theano.config.floatX))
    rstate = rstream.new_shared_rstate()

    new_rstate, out = tensor.raw_random.multinomial(rstate, draw_shape, n, p, dtype=dtype)
    rstream.add_default_update(out, rstate, new_rstate)
    return out

def logFactorial(x):
    return logGamma(x+1.)
    
@rng_register
def multinomial_lpdf(node, x, kw):
    r, shape, n, p = node.inputs

    # TODO: how do I check this ?
    # assert n == tensor.sum(x)
        
    x = tensor.as_tensor_variable(x).astype(theano.config.floatX)    
    
    return logFactorial(n) - tensor.sum(logFactorial(x), axis=1) + tensor.sum(tensor.log(p)*x, axis=1)

@rng_register
def multinomial_proposal(rstream, node, sample, kw):
    return node.outputs[1]

# some weirdness because raw_random uses a helper function
# TODO: is there a clear way to fix this ?
@rng_register
def multinomial_helper_sampler(*args, **kwargs):
    return multinomial_sampler(*args, **kwargs)
    
@rng_register
def multinomial_helper_lpdf(*args, **kwargs):
    return multinomial_lpdf(*args, **kwargs)

@rng_register
def multinomial_helper_proposal(*args, **kwargs):
    return multinomial_proposal(*args, **kwargs)

# ---------
# Dirichlet-Multinomial
#
# Only the LPDF is implemented, the sampler is bogus
# ---------

class DM(theano.Op):
    dist_name = 'DM'
    def __init__(self, otype):
        self.otype = otype
    
    def make_node(self, s_rstate, alpha):
        alpha = tensor.as_tensor_variable(alpha)
        return theano.gof.Apply(self,
                [s_rstate, alpha],
                [s_rstate.type(), self.otype()])

    def perform(self, node, inputs, output_storage):      
        raise NotImplemented
        
@rng_register
def DM_sampler(rstream, alpha, draw_shape=None, ndim=None, dtype=None):
    shape = infer_shape(rstream.dirichlet(alpha, draw_shape=draw_shape))
    rstate = rstream.new_shared_rstate()
    op = DM(tensor.TensorType(broadcastable=(False,)* tensor.get_vector_length(shape), dtype=theano.config.floatX))
    rs, out = op(rstate, alpha)
    return out
    
@rng_register
def DM_lpdf(node, sample, kw):
    r, alpha = node.inputs
    return logBeta(sample + alpha) - logBeta(alpha)

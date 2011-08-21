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
from for_theano import elemwise_cond
from for_theano import ancestors
from rstreams import rng_register

# TODOs:
# - Additional distributions of interest:
#   - Multinomial
#   - Wishart
#   - Dirichlet process / CRP
# - Proposal distributions

# -------
# Uniform
# -------


@rng_register
def uniform_sampler(rstream, low=0.0, high=1.0, ndim=None, draw_shape=None, dtype=theano.config.floatX):
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
def normal_sampler(rstream, mu=0.0, sigma=1.0, draw_shape=None, ndim=0, dtype=None):
    if not isinstance(mu, theano.Variable):
        mu = tensor.shared(numpy.asarray(mu, dtype=theano.config.floatX))
    if not isinstance(sigma, theano.Variable):
        sigma = tensor.shared(numpy.asarray(sigma, dtype=theano.config.floatX))
    rstate = rstream.new_shared_rstate()

    # James: why is this required? fails in draw_shape is not provided
    # if isinstance(draw_shape, (list, tuple)):
    #     draw_shape = tensor.stack(*draw_shape)

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
def lognormal_sampler(s_rstate, mu=0.0, sigma=1.0, shape=None, ndim=None, dtype=theano.config.floatX):
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
    ndim, shape, bcast = tensor.raw_random._infer_ndim_bcast(
            ndim, shape, mu, sigma)
    op = tensor.raw_random.RandomFunction('lognormal',
            tensor.TensorType(dtype=dtype, broadcastable=bcast))
    return op(s_rstate, shape, mu, sigma)


@rng_register
def lognormal_lpdf(node, x, kw):
    r, shape, mu, sigma = node.inputs
    Z = sigma * x * numpy.sqrt(2 * numpy.pi)
    E = 0.5 * ((tensor.log(x) - mu) / sigma)**2
    return -E - tensor.log(Z)


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

    def make_node(self, s_rstate, p):
        p = tensor.as_tensor_variable(p)
        if p.ndim != 1: raise NotImplementedError()
        return theano.gof.Apply(self,
                [s_rstate, p],
                [s_rstate.type(), self.otype()])

    def perform(self, node, inputs, outstor):
        rng, p = inputs
        if not self.destructive:
            rng = copy.deepcopy(rng)
        counts = rng.multinomial(pvals=p, n=1)
        oval = numpy.where(counts)[0][0]
        outstor[0][0] = rng
        outstor[1][0] = self.otype.filter(oval, allow_downcast=True)


@rng_register
def categorical_sampler(rstate, p, shape=None, ndim=None, dtype='int32'):
    if shape != None:
        raise NotImplementedError()
    op = Categorical(False,
            tensor.TensorType(
                broadcastable=(),
                dtype=dtype))
    return op(rstate, p)


@rng_register
def categorical_lpdf(node, sample, kw):
    """
    Return a random integer from 0 .. N-1 inclusive according to the
    probabilities p[0] .. P[N-1].

    This is formally equivalent to numpy.where(multinomial(n=1, p))
    """
    # WARNING: I think the p[-1] is not used, but assumed to be p[:-1].sum()
    s_rstate, p = node.inputs
    return p[sample]


# ---------
# LogGamma helper Op
# ---------

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
    output_storage[0][0] = scipy.special.gammaln(x)

logGamma = LogGamma()

# ---------
# Dirichlet
# ---------

@rng_register
def dirichlet_sampler(rstream, alpha, draw_shape=None, ndim=None, dtype=theano.config.floatX):
    tmp = alpha.T[0].T

    alpha = tensor.as_tensor_variable(alpha)
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

    return -logBeta(alpha) + tensor.sum(tensor.log(sample)*(alpha-1))
    
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
    r, shape, a, b = node.inputs

    return (a-1)*tensor.log(x) - x/b - logGamma(a) - a*tensor.log(b)
    
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
    return logGamma(x+1)
    
@rng_register
def multinomial_lpdf(node, x, kw):
    r, shape, n, p = node.inputs

    # TODO: how do I check this ?
    # assert n == tensor.sum(x)
    
    return logFactorial(n) - tensor.sum(logFactorial(x)) + tensor.sum(tensor.log(p)*x)

# some weirdness because raw_random uses a helper function
# TODO: is there a clear way to fix this ?
@rng_register
def multinomial_helper_sampler(*args, **kwargs):
    return multinomial_sampler(*args, **kwargs)
    
@rng_register
def multinomial_helper_lpdf(*args, **kwargs):
    return multinomial_lpdf(*args, **kwargs)
    
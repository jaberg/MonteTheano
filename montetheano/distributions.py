"""
Math for various distributions.

"""
import __builtin__
import copy
import numpy
import theano
from theano import tensor
from for_theano import elemwise_cond
from for_theano import ancestors
from rstreams import rng_register

# -------
# Uniform
# -------


@rng_register
def uniform_sampler(rstream, shape=None, low=0.0, high=1.0, ndim=None, dtype=None):
    rstate = rstream.new_shared_rstate()
    new_rstate, out = tensor.raw_random.uniform(rstate, shape, low, high, ndim, dtype)
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
def normal_sampler(rstream, shape=None, mu=0.0, sigma=1.0, ndim=0, dtype=None):
    if not isinstance(mu, theano.Variable):
        mu = tensor.shared(numpy.asarray(mu, dtype=theano.config.floatX))
    if not isinstance(sigma, theano.Variable):
        sigma = tensor.shared(numpy.asarray(sigma, dtype=theano.config.floatX))
    rstate = rstream.new_shared_rstate()

    print rstate, shape, mu, sigma, dtype
    
    new_rstate, out = tensor.raw_random.normal(rstate, shape, mu, sigma, dtype=dtype)
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
def binomial_sampler(rstream, shape=None, n=1, p=0.5, ndim=0, dtype=None):
    if not isinstance(n, theano.Variable):
        n = tensor.shared(numpy.asarray(n, dtype=int))
    if not isinstance(p, theano.Variable):
        p = tensor.shared(numpy.asarray(p, dtype=theano.config.floatX))
    rstate = rstream.new_shared_rstate()
    new_rstate, out = tensor.raw_random.binomial(rstate, shape, n, p, dtype=dtype)
    rstream.add_default_update(out, rstate, new_rstate)
    return out

@rng_register
def binomial_lpdf(rv, sample, kw):
    random_state, size, n, p = rv.owner.inputs

    # for the n > 1 the "choose" operation is required
    # TODO assert n == 1
    
    return tensor.switch(tensor.eq(sample, 1.), tensor.log(p), tensor.log(1. - p))

@rng_register
def binomial_params(node):
    rstate, shape, n, p = node.inputs
    return [n, p]

# ---------
# LogNormal
# ---------


@rng_register
def lognormal_sampler(s_rstate, mu=0.0, sigma=1.0, shape=None, ndim=None, dtype=None):
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
# Dirichlet
# ---------


class Dirichlet(theano.Op):
    dist_name = 'dirichlet'
    def __init__(self, destructive):
        self.destructive = destructive
        if destructive:
            self.destroy_map = {0:[0]}
        else:
            self.destroy_map = {}

    def __eq__(self, other):
        return (type(self) == type(other)
                and self.destructive == other.destructive)

    def __hash__(self):
        return hash((type(self), self.destructive))

    def make_node(self, s_rstate, alpha):
        alpha = tensor.as_tensor_variable(alpha)
        if alpha.ndim != 1: raise NotImplementedError()
        return theano.gof.Apply(self,
                [s_rstate, alpha],
                [s_rstate.type(), alpha.type()])

    def perform(self, node, inputs, outstor):
        rng, alpha = inputs
        if not self.destructive:
            rng = copy.deepcopy(rng)
        oval = rng.dirichlet(alpha=alpha).astype(alpha.dtype)
        outstor[0][0] = rng
        outstor[1][0] = oval


@rng_register
def dirichlet_sampler(rstate, alpha, shape=None, ndim=None, dtype=None):
    if shape != None:
        raise NotImplementedError()
    if dtype != None:
        raise NotImplementedError()
    op = Dirichlet(False)
    return op(rstate, alpha)


@rng_register
def dirichlet_lpdf(node, sample, kw):
    """

    http://en.wikipedia.org/wiki/Dirichlet_distribution

    """
    raise NotImplementedError()


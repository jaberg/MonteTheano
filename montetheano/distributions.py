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
def uniform_sampler(rstream, low=0.0, high=1.0, shape=None, ndim=None, dtype=None):
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
def normal_sampler(rstream, mu=0.0, sigma=1.0, shape=None, ndim=0, dtype=None):
    if not isinstance(mu, theano.Variable):
        mu = tensor.shared(numpy.asarray(mu, dtype=theano.config.floatX))
    if not isinstance(mu, theano.Variable):
        sigma = tensor.shared(numpy.asarray(sigma, dtype=theano.config.floatX))
    rstate = rstream.new_shared_rstate()
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

def full_sample(s_rng, outputs ):
    all_vars = ancestors(outputs)
    assert outputs[0] in all_vars
    RVs = [v for v in all_vars if is_random_var(v)]
    rdict = dict([(v, v) for v in RVs])

    if True:
        # outputs is same
        raise NotImplementedError()
    elif isinstance(size, int):
        # use scan
        raise NotImplementedError()
    else:
        n_steps = theano.tensor.prod(size)
        # use scan for n_steps
        #scan_outputs = ...
        outputs = scan_outputs[:len(outputs)]
        s_RVs = scan_outputs[len(outputs):]
        # reshape so that leading dimension goes from n_steps -> size
        raise NotImplementedError()
    return outputs, rdict

# Sample the generative model and return "outputs" for cases where "condition" is met.
# If no condition is given, it just samples from the model
# The outputs can be a single TheanoVariable or a list of TheanoVariables.
# The function returns a single sample or a list of samples, depending on "outputs"; and an updates dictionary.
def rejection_sample(outputs, condition = None):
    if isinstance(outputs, tensor.TensorVariable):
        init = [0]
    else:
        init = [0]*len(outputs)
    if condition is None:
        # TODO: I am just calling scan to get updates, can't I create this myself?
        # output desired RVs when condition is met
        def rejection():
            return outputs

        samples, updates = theano.scan(rejection, outputs_info = init, n_steps = 1)
    else:
        # output desired RVs when condition is met
        def rejection():
            return outputs, {}, theano.scan_module.until(condition)
        samples, updates = theano.scan(rejection, outputs_info = init, n_steps = 1000)
    if isinstance(samples, tensor.TensorVariable):
        sample = samples[-1]
    else:
        sample = [s[-1] for s in samples]
    return sample, updates


@rng_register
def normal_params(node):
    rstate, shape, mu, sigma = node.inputs
    return [mu, sigma]

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
def categorical_pdf(node, sample, kw):
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
def dirichlet_pdf(node, sample, kw):
    """

    http://en.wikipedia.org/wiki/Dirichlet_distribution

    """
    raise NotImplementedError()

def mh_sample(s_rng, outputs, observations = {}):
    # TODO: should there be a size variable here?
    # TODO: implement lag and burn-in
    # TODO: implement size
    """
    Return a dictionary mapping random variables to their sample values.
    """

    all_vars = ancestors(list(outputs) + list(observations.keys()))
    for o in observations:
        assert o in all_vars
        if not is_random_var(o):
            raise TypeError(o)

    free_RVs = [v for v in RVs if v not in observations]

    # TODO: sample from the prior to initialize these guys?
    # free_RVs_state = [theano.shared(v) for v in free_RVs]
    # TODO: how do we infer shape?
    free_RVs_state = [theano.shared(0.5*numpy.ones(shape=())) for v in free_RVs]
    free_RVs_prop = [s_rng.normal(size = (), std = .1) for v in free_RVs]

    log_likelihood = theano.shared(numpy.array(float('-inf')))

    U = s_rng.uniform(size=(), low=0, high=1.0)

    # TODO: can we pre-generate the noise
    def mcmc(ll, *frvs):
        # TODO: implement generic proposal distributions
        # TODO: how do we infer shape?
        proposals = [(rvs + rvp) for rvs,rvp in zip(free_RVs_state, free_RVs_prop)]

        full_observations = dict(observations)
        full_observations.update(dict([(rv, s) for rv, s in zip(free_RVs, proposals)]))

        new_log_likelihood = full_log_likelihood(observations = full_observations)

        accept = tensor.or_(new_log_likelihood > ll, U <= tensor.exp(new_log_likelihood - ll))

        return [tensor.switch(accept, new_log_likelihood, ll)] + \
               [tensor.switch(accept, p, f) for p, f in zip(proposals, frvs)], \
               {}, theano.scan_module.until(accept)

    samples, updates = theano.scan(mcmc, outputs_info = [log_likelihood] + free_RVs_state, n_steps = 10000000)
    updates[log_likelihood] = samples[0][-1]
    updates.update(dict([(f, s[-1]) for f, s in zip(free_RVs_state, samples[1:])]))
    
    return [free_RVs_state[free_RVs.index(out)] for out in outputs], log_likelihood, updates

def hybridmc_sample(s_rng, outputs, observations = {}):
    # TODO: should there be a size variable here?
    # TODO: implement lag and burn-in
    # TODO: implement size
    """
    Return a dictionary mapping random variables to their sample values.
    """

    all_vars = ancestors(list(outputs) + list(observations.keys()))
    
    for o in observations:
        assert o in all_vars
        if not is_random_var(o):
            raise TypeError(o)

    RVs = [v for v in all_vars if is_random_var(v)]

    free_RVs = [v for v in RVs if v not in observations]
    
    free_RVs_state = [theano.shared(0.5*numpy.ones(shape=())) for v in free_RVs]    
    free_RVs_prop = [s_rng.normal(size = (), std = 1) for v in free_RVs]    
    
    log_likelihood = theano.shared(numpy.array(float('-inf')))
    
    U = s_rng.uniform(size=(), low=0, high=1.0)
    
    epsilon = numpy.sqrt(2*0.03)
    def mcmc(ll, *frvs):
        full_observations = dict(observations)
        full_observations.update(dict([(rv, s) for rv, s in zip(free_RVs, frvs)]))
        loglik = -full_log_likelihood(observations = full_observations)

        proposals = free_RVs_prop
        H = tensor.add(*[tensor.sum(tensor.sqr(p)) for p in proposals])/2. + loglik

# -- this should be an inner loop
        g = tensor.grad(loglik, frvs)
        proposals = [(p - epsilon*g/2.) for p, g in zip(proposals, g)]

        rvsp = [(rvs + epsilon*rvp) for rvs,rvp in zip(frvs, proposals)]
        
        full_observations = dict(observations)
        full_observations.update(dict([(rv, s) for rv, s in zip(free_RVs, rvsp)]))
        new_loglik = -full_log_likelihood(observations = full_observations)
        
        gnew = tensor.grad(new_loglik, rvsp)
        proposals = [(p - epsilon*gn/2.) for p, gn in zip(proposals, gnew)]
# --
        
        Hnew = tensor.add(*[tensor.sum(tensor.sqr(p)) for p in proposals])/2. + new_loglik

        dH = Hnew - H
        accept = tensor.or_(dH < 0., U < tensor.exp(-dH))

        return [tensor.switch(accept, -new_loglik, ll)] + \
            [tensor.switch(accept, p, f) for p, f in zip(rvsp, frvs)], \
            {}, theano.scan_module.until(accept)

    samples, updates = theano.scan(mcmc, outputs_info = [log_likelihood] + free_RVs_state, n_steps = 10000000)
    
    updates[log_likelihood] = samples[0][-1]
    updates.update(dict([(f, s[-1]) for f, s in zip(free_RVs_state, samples[1:])]))
    
    return [free_RVs_state[free_RVs.index(out)] for out in outputs], log_likelihood, updates


# ----------
# Undirected
# ----------

class Undirected(theano.Op):
    def __init__(self, ops, n_inputs):
        self.ops = ops
        self.n_inputs = n_inputs



# UNVERIFIED
class Normal(RV):
    def __init__(self, mu, sigma):
        self.mu = as_rv_or_sharedX(mu)
        self.sigma = as_rv_or_sharedX(sigma)

    def sample(self, draw_shape, rstreams, sample):
        mu = sample(self.mu, ())
        sigma = sample(self.sigma, ())
        return rstreams.normal(draw_shape, mu, sigma)

    def pdf(self, x):
        one = tensor.as_tensor_variable(numpy.asarray(1, dtype='float32'))
        rstate, shape, mu, sigma = node.inputs
        Z = tensor.sqrt(2 * numpy.pi * sigma**2)
        E = 0.5 * ((mu - x)/(one*sigma))**2
        return tensor.exp(-E) / Z

    def params(self):
        return [i for i in [self.mu, self.sigma]
                if isinstance(i, theano.Variable)]

    def posterior(self, x, weights=None):
        """
        Message-passing required.
        """
        raise NotImplementedError() 

    def maximum_likelihood(self, x, weights=None):
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
        rval = Updates()
        if self.mu in self.params():
            rval[self.mu] = new_mu
        if self.sigma in self.params():
            rval[self.sigma] = new_sigma
        return rval


# UNVERIFIED
class Categorical(RV):
    def __init__(self, weights):
        self.weights = weights

    def sample(self, N, rstreams):
        raise NotImplementedError()

    def pdf(self, x):
        raise NotImplementedError()


# UNVERIFIED
class List(RV):
    def __init__(self, components):
        self.components = components

    def __getitem__(self, idx):
        if isinstance(idx, Categorical):
            return Mixture(Categorical.weights, self.components)
        else:
            return self.components[idx]


# UNVERIFIED
class Dict(RV):
    def __init__(self, **kwargs):
        self.components = kwargs

    def sample(self, draw_shape, rstreams, memo):
        raise NotImplementedError()

    def pdf(self, X):


# UNVERIFIED
class Mixture(RV):
    def __init__(self, weights, components):
        self.weights = weights
        self.components = components

    def sample(self, draw_shape, rstreams, memo):
        raise NotImplementedError()

    def pdf(self, x):
        raise NotImplementedError()


# UNVERIFIED
@register_pdf
def binomial(rv, sample, kw):
    if (rv.owner
            and isinstance(rv.owner.op, tensor.raw_random.RandomFunction)
            and rv.owner.op.fn == numpy.random.RandomState.binomial):
        random_state, size, n, p = rv.owner.inputs

        # for the n > 1 the "choose" operation is required
        # TODO assert n == 1
        
        return tensor.switch(tensor.eq(sample, 1.), tensor.log(p), tensor.log(1. - p))
    else:
        raise WrongPdfHandler()


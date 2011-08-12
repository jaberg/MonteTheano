"""
"""
import __builtin__
import copy

import numpy

import theano
from theano import tensor

class RandomStreams(object):
    samplers = {}
    pdfs = {}
    local_proposals = {}

    def __init__(self, seed):
        self.state_updates = []
        self.default_instance_seed = seed
        self.seed_generator = numpy.random.RandomState(seed)

    def sample(self, dist_name, *args, **kwargs):
        handler = self.samplers[dist_name]

        seed = int(self.seed_generator.randint(2**30))
        random_state_variable = theano.shared(
                numpy.random.RandomState(seed))
        new_r, out = handler(random_state_variable, *args, **kwargs)
        out.rng = random_state_variable
        out.update = (random_state_variable, new_r)
        self.state_updates.append(out.update)
        random_state_variable.default_update = new_r
        return out

    def seed(self, seed=None):
        """Re-initialize each random stream

        :param seed: each random stream will be assigned a unique state that depends
        deterministically on this value.

        :type seed: None or integer in range 0 to 2**30

        :rtype: None
        """
        if seed is None:
            seed = self.default_instance_seed

        seedgen = numpy.random.RandomState(seed)
        for old_r, new_r in self.state_updates:
            old_r_seed = seedgen.randint(2**30)
            old_r.set_value(numpy.random.RandomState(int(old_r_seed)),
                    borrow=True)

    def pdf(self, rv, sample, **kwargs):
        """
        Return the probability (density) that random variable `rv`, returned by
        a call to one of the sampling routines of this class takes value `sample`
        """
        _sample = theano.tensor.as_tensor_variable(sample)
        if rv.owner:
            try:
                dist_name = rv.owner.op.dist_name
            except AttributeError:
                try:
                    dist_name = rv.owner.op.fn.__name__
                except AttributeError:
                    raise TypeError('rv not recognized as output of RandomFunction')
            pdf = self.pdfs[dist_name]
            return pdf(rv.owner, _sample, **kwargs)
        else:
            raise TypeError('rv not recognized as output of RandomFunction')

    def local_proposal(rv, sample, **kwargs):
        """
        Return the probability (density) that random variable `rv`, returned by
        a call to one of the sampling routines of this class takes value `sample`
        """
        raise NotImplementedError()

    #
    # N.B. OTHER METHODS (samplers) ARE INSTALLED HERE BY
    # - register_sampler
    # - rng_register
    #


def register_sampler(dist_name, f):
    """
    Inject a sampling function into RandomStreams for the distribution with name
    f.__name__
    """
    # install an instancemethod on the RandomStreams class
    # that is a shortcut for something like
    # self.sample('uniform', *args, **kwargs)

    def sampler(self, *args, **kwargs):
        return self.sample(dist_name, *args, **kwargs)
    setattr(RandomStreams, dist_name, sampler)

    if dist_name in RandomStreams.samplers:
        # TODO: allow for multiple handlers?
        raise KeyError(dist_name)
    RandomStreams.samplers[dist_name] = f
    return f


def register_pdf(dist_name, f):
    if dist_name in RandomStreams.pdfs:
        # TODO: allow for multiple handlers?
        raise KeyError(dist_name)
    RandomStreams.pdfs[dist_name] = f
    return f


def rng_register(f):
    if f.__name__.endswith('_sampler'):
        dist_name = f.__name__[:-len('_sampler')]
        return register_sampler(dist_name, f)

    elif f.__name__.endswith('_pdf'):
        dist_name = f.__name__[:-len('_pdf')]
        return register_pdf(dist_name, f)
    else:
        raise ValueError("function name doesn't end with _sampler or _pdf")


class ClobberContext(object):
    def __enter__(self):
        not hasattr(self, 'old')
        self.old = {}
        for name in self.clobber:
            if hasattr(__builtin__, name):
                self.old[name] = getattr(__builtin__, name)
            if hasattr(self.obj, name):
                setattr(__builtin__, name, getattr(self.obj, name))
        return self.obj

    def __exit__(self, e_type, e_val, e_traceback):
        for name in self.clobber:
            if name in self.old:
                setattr(__builtin__, name, self.old[name])
            elif hasattr(__builtin__, name):
                delattr(__builtin__, name)
        del self.old


class srng_globals(ClobberContext):
    def __init__(self, obj):
        if isinstance(obj, int):
            self.obj = RandomStreams(23424)
        else:
            self.obj = obj
        self.clobber = self.obj.samplers.keys()


#####################
# Stock distributions
#####################


# -------
# Uniform
# -------


@rng_register
def uniform_sampler(rstate, low=0.0, high=0.0, shape=None, ndim=None, dtype=None):
    return tensor.raw_random.uniform(rstate, shape, low, high, ndim, dtype)


@rng_register
def uniform_pdf(node, sample):
    # make sure that the division is done at least with float32 precision
    one = tensor.as_tensor_variable(numpy.asarray(1, dtype='float32'))
    rval = elemwise_cond(
        0,                sample < low,
        one / (high-low), sample <= high,
        0)
    return rval


# ------
# Normal
# ------


@rng_register
def normal_sampler(rstate, mu=0.0, sigma=1.0, shape=None, ndim=0, dtype=None):
    return tensor.raw_random.normal(rstate, shape, mu, sigma, dtype=dtype)


@rng_register
def normal_pdf(node, sample, kw):
    # make sure that the division is done at least with float32 precision
    one = tensor.as_tensor_variable(numpy.asarray(1, dtype='float32'))
    Z = tensor.sqrt(2 * numpy.pi * std**2)
    E = 0.5 * ((avg - sample)/(one*std))**2
    return tensor.exp(-E) / Z


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
def lognormal_pdf(node, x, kw):
    r, shape, mu, sigma = node.inputs
    Z = sigma * x * numpy.sqrt(2 * numpy.pi)
    E = 0.5 * ((tensor.log(x) - mu) / sigma)**2
    return tensor.exp(-E)/Z


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


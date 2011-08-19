"""
Registry and definition for new-and-improved RandomStreams
"""

import copy
import numpy
import theano
from theano import tensor
from for_theano import elemwise_cond
from for_theano import ancestors
from utils import ClobberContext

samplers = {}
pdfs = {}
ml_handlers = {}
params_handlers = {}
local_proposals = {}
randomstate_types = (tensor.raw_random.RandomStateType,)


def rv_dist_name(rv):
    try:
        return rv.owner.op.dist_name
    except AttributeError:
        try:
            return rv.owner.op.fn.__name__
        except AttributeError:
            raise TypeError('rv not recognized as output of RandomFunction')

class RandomStreams(ClobberContext):
    clobber_symbols = ['pdf']

    def __init__(self, seed):
        self.state_updates = []
        self.default_instance_seed = seed
        self.seed_generator = numpy.random.RandomState(seed)
        self.default_updates = {}

    def shared(self, val, **kwargs):
        rval = theano.shared(val, **kwargs)
        return rval

    def sharedX(self, val, **kwargs):
        rval = theano.shared(
                numpy.asarray(val, dtype=theano.config.floatX),
                **kwargs)
        return rval

    def new_shared_rstate(self):
        seed = int(self.seed_generator.randint(2**30))
        rval = theano.shared(numpy.random.RandomState(seed))
        return rval

    def add_default_update(self, used, recip, new_expr):
        if used not in self.default_updates:
            self.default_updates[used] = {}
        self.default_updates[used][recip] = new_expr
        used.update = (recip, new_expr) # necessary?
        recip.default_update = new_expr
        self.state_updates.append((recip, new_expr))

    def sample(self, dist_name, *args, **kwargs):
        handler = samplers[dist_name]
        out = handler(self, *args, **kwargs)
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
        if rv.owner:
            dist_name = rv_dist_name(rv)
            pdf = pdfs[dist_name]
            return pdf(rv.owner, sample, kwargs)
        else:
            raise TypeError('rv not recognized as output of RandomFunction')

    def ml(self, rv, sample, weights=None):
        """
        Return an Updates object mapping distribution parameters to expressions
        of their maximum likelihood values.
        """
        if rv.owner:
            dist_name = rv_dist_name(rv)
            pdf = ml_handlers[dist_name]
            return pdf(rv.owner, sample, weights=weights)
        else:
            raise TypeError('rv not recognized as output of RandomFunction')

    def params(self, rv):
        """
        Return an Updates object mapping distribution parameters to expressions
        of their maximum likelihood values.
        """
        if rv.owner:
            return self.params_handlers[rv_dist_name(rv)](rv.owner)
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
    RandomStreams.clobber_symbols.append(dist_name)

    if dist_name in samplers:
        # TODO: allow for multiple handlers?
        raise KeyError(dist_name)
    samplers[dist_name] = f
    return f


def register_lpdf(dist_name, f):
    if dist_name in pdfs:
        # TODO: allow for multiple handlers?
        raise KeyError(dist_name, pdfs[dist_name])
    pdfs[dist_name] = f
    return f


#TODO: think about what this function is supposed to do??
def register_ml(dist_name, f):
    if dist_name in ml_handlers:
        # TODO: allow for multiple handlers?
        raise KeyError(dist_name, ml_handlers[dist_name])
    ml_handlers[dist_name] = f
    return f


#TODO: think about what this function is supposed to do??
def register_params(dist_name, f):
    if dist_name in params_handlers:
        # TODO: allow for multiple handlers?
        raise KeyError(dist_name, params_handlers[dist_name])
    params_handlers[dist_name] = f
    return f


def rng_register(f):
    if f.__name__.endswith('_sampler'):
        dist_name = f.__name__[:-len('_sampler')]
        return register_sampler(dist_name, f)

    elif f.__name__.endswith('_lpdf'):
        dist_name = f.__name__[:-len('_lpdf')]
        return register_lpdf(dist_name, f)

    elif f.__name__.endswith('_ml'):
        dist_name = f.__name__[:-len('_ml')]
        return register_ml(dist_name, f)

    elif f.__name__.endswith('_params'):
        dist_name = f.__name__[:-len('_params')]
        return register_params(dist_name, f)

    else:
        raise ValueError("function name suffix not recognized", f.__name__)


def lpdf(rv, sample):
    """
    Return the probability (density) that random variable `rv`, returned by
    a call to one of the sampling routines of this class takes value `sample`
    """
    if not is_rv(rv):
        raise TypeError('rv not recognized as a random variable', rv)

    if is_raw_rv(rv):
        dist_name = rv_dist_name(rv)
        pdf = pdfs[dist_name]
        return pdf(rv.owner, sample, kwargs)
    else:
        #TODO: infer from the ancestors of v what distribution it
        #      has.
        raise NotImplementedError()


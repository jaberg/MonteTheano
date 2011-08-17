"""
Registry of probability density functions (PDFs)
"""
import numpy
import theano
from theano import tensor
from theano.gof import graph

from for_theano import ancestors

from for_theano import multiswitch
from shallow_clone import clone_keep_replacements

_pdf_handlers = []
_proposal_handlers = []


class WrongPdfHandler(Exception):
    pass


def is_random_var(v):
    """
    Return True iff v is a Random Variable

    """
    #TODO: How to make this work with non-standard RandomStreams?
    if v.owner and isinstance(v.owner.op, tensor.raw_random.RandomFunction):
        return True
    return False

def RVs(outputs):
    """
    Return a list of all random variables required to compute `outputs`.
    """
    all_vars = ancestors(outputs)
    assert outputs[0] in all_vars
    rval = [v for v in all_vars if is_random_var(v)]
    return rval


def register_pdf(f):
    _pdf_handlers.append(f)
    return f

def register_proposal(f):
    _proposal_handlers.append(f)
    return f


def log_pdf(rv, sample, **kwargs):
    """
    Return the probability (density) that `rv` takes value `sample`
    """
    _sample = theano.tensor.as_tensor_variable(sample)
    for handler in _pdf_handlers:
        try:
            return handler(rv, _sample, kwargs)
        except WrongPdfHandler:
            continue
    raise TypeError('unrecognized random variable', rv)

def proposal(rv):
    for handler in _proposal_handlers:
        try:
            return handler(rv)
        except WrongPdfHandler:
            continue
    raise TypeError('unrecognized random variable', rv)


def full_log_likelihood(observations, keep_unobserved=False):
    """
    \sum_i log(P(observations)) given that observations[i] ~ RV, iid.

    observations: a dictionary mapping random variables to tensors.

        observations[RV] = rv_observations

        rv_observations[i] is the i'th observation or RV

    """
    
    RVs = [v for v in ancestors(observations.keys()) if is_random_var(v)]
    for rv in RVs:
        if rv not in observations:
            if keep_unobserved:
                observations[rv] = rv
            else:
                raise ValueError('missing observations')

    # Ensure we can work on tensor variables:
    observations = dict([(rv, tensor.as_tensor_variable(obs).astype(rv.dtype)) for rv, obs in observations.items()])
            
    pdfs = [log_pdf(rv, obs) for rv,obs in observations.items()]

    lik = tensor.add(*[tensor.sum(p) for p in pdfs])

    dfs_variables = ancestors([lik], blockers=RVs)
    frontier = [r for r in dfs_variables if r.owner is None or r in RVs]
    cloned_inputs, cloned_outputs = clone_keep_replacements(frontier, [lik], replacements=dict(observations.items()))
    cloned_lik, = cloned_outputs

    return cloned_lik
    
###########################################################
# Stock pdfs for distributions in theano.tensor.raw_random
###########################################################

@register_pdf
def uniform(rv, sample, kw):
    if (rv.owner
            and isinstance(rv.owner.op, tensor.raw_random.RandomFunction)
            and rv.owner.op.fn == numpy.random.RandomState.uniform):
        random_state, size, low, high = rv.owner.inputs

        # make sure that the division is done at least with float32 precision
        one = tensor.as_tensor_variable(numpy.asarray(1, dtype='float32'))
        rval = multiswitch(
            numpy.array(float('-inf')), sample < low,
            -tensor.log(high-low), sample <= high,
            numpy.array(float('-inf')))
        return rval
    else:
        raise WrongPdfHandler()


@register_pdf
def normal(rv, sample, kw):
    if (rv.owner
            and isinstance(rv.owner.op, tensor.raw_random.RandomFunction)
            and rv.owner.op.fn == numpy.random.RandomState.normal):
        random_state, size, avg, std = rv.owner.inputs
        
        # make sure that the division is done at least with float32 precision
        one = tensor.as_tensor_variable(numpy.asarray(1, dtype='float32'))
        Z = tensor.sqrt(2. * numpy.pi * std**2)
        E = -(sample - avg)**2./(2.*(one*std)**2.)
        return E - tensor.log(Z)
    else:
        raise WrongPdfHandler()


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

# @register_proposal
# def discrete(rv):
#     if (rv.owner
#             and isinstance(rv.owner.op, tensor.raw_random.RandomFunction)
#             and rv.owner.op.fn == numpy.random.RandomState.binomial):
# 
#         return rv.clone()
#     else:
#         raise WrongPdfHandler()
    
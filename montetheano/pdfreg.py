"""
Registry of probability density functions (PDFs)
"""
import numpy
import theano
from theano import tensor

from for_theano import multiswitch


_pdf_handlers = []


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


def register_pdf(f):
    _pdf_handlers.append(f)
    return f


def pdf(rv, sample, **kwargs):
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


def full_likelihood(observations):
    """
    \sum_i log(P(observations)) given that observations[i] ~ RV, iid.

    observations: a dictionary mapping random variables to tensors.

        observations[RV] = rv_observations

        rv_observations[i] is the i'th observation or RV

    """
    RVs = [v for v in ancestors(observations.keys()) if is_random_var(v)]
    for rv in RVs:
        if rv not in observations:
            raise ValueError('missing observations')
    pdfs = [pdf(rv, obs) for rv,obs in observations.items()]

    lik = tensor.mul(*[tensor.prod(p) for p in pdfs])

    cloned_inputs, cloned_outputs, otherstuff = rebuild_collect_shared(
            outputs=[lik],
            replace=observations,
            copy_inputs_over=False,
            no_default_updates=True)

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
            0,                sample < low,
            one / (high-low), sample <= high,
            0)
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
        Z = tensor.sqrt(2 * numpy.pi * std**2)
        E = 0.5 * ((avg - sample)/(one*std))**2
        return tensor.exp(-E) / Z
    else:
        raise WrongPdfHandler()


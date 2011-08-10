"""
Registry of probability density functions (PDFs)
"""
import numpy
import theano
from theano import tensor


_pdf_handlers = []


class WrongPdfHandler(Exception):
    pass


def is_random_var(v):
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
    if not is_random_var(rv):
        raise TypeError(rv)
    _sample = theano.tensor.as_tensor_variable(sample)
    for handler in _pdf_handlers:
        try:
            return handler(rv, _sample, kwargs)
        except WrongPdfHandler:
            continue
    raise WrongPdfHandler('no handler found')


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


# HELPER FUNCTIONS THAT MIGHT GET INTO THEANO?
def multiswitch(*args):
    """Build a nested elemwise if elif ... statement.

        multiswitch(
            a, cond_a,
            b, cond_b,
            c)

    Translates roughly to an elementwise version of this...

        if cond_a:
            a
        elif cond_b:
            b
        else:
            c
    """
    assert len(args) % 2, 'need an add number of args'
    if len(args) == 1:
        return args[0]
    else:
        return tensor.switch(
                args[1],
                args[0],
                multiswitch(*args[2:]))

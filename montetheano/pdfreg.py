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
def normal(rv, sample, kw):
    if (rv.owner
            and isinstance(rv.owner.op, tensor.raw_random.RandomFunction)
            and rv.owner.op.fn == numpy.random.RandomState.normal):
        random_state, size, avg, std = rv.owner.inputs
        # make sure that the division is done at least with float32 precision
        f32 = tensor.as_tensor_variable(numpy.asarray(1, dtype='float32'))
        Z = tensor.sqrt(2 * numpy.pi * std**2)
        E = 0.5 * ((avg - sample)/(f32*std))**2
        return tensor.prod(tensor.exp(-E) / Z)
    else:
        raise WrongPdfHandler()

#@register_pdf
def uniform(rv, sample, kw):
    random_state, size, avg, std = rv.owner.inputs
    # assume sample has sample.shape == size
    Z = (2 * numpy.pi * std**2)
    E = 0.5 * ((avg - sample)/(f32*std))**2
    return tensor.prod(tensor.exp(-E) / Z)

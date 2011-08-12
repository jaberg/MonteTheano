"""
Registry of probability density functions (PDFs)

These handlers will work on any Theano variable whose owner.op has a `.fn`
attribute that is in the _pdf_handlers registry.

TODO: add the `.fn` to the MRG and GPU random number generators so that they are
recognized by this registry system.

"""
import numpy
import theano
from theano import tensor

from theano.compile import rebuild_collect_shared
from theano.gof.graph import ancestors
from theano.tensor.raw_random import RandomFunction

from for_theano import elemwise_cond




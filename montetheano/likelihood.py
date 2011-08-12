import sys
import numpy
import theano
from theano import tensor

from theano.compile import rebuild_collect_shared
from theano.gof.graph import ancestors

from for_theano import multiswitch

switch_msg = """
Warning: full_likelihood can be incorrect in the case of graphs with
switch or ifelse.  It might also be incorrect in all cases.  Must think about
this to ensure this function has the right semantics.
"""

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



def full_likelihood(outputs, observations):
    """
    \sum_i log(P(observations)) given that observations[i] ~ RV, iid.

    observations: a dictionary mapping random variables to tensors.

        observations[RV] = rv_observations

        rv_observations[i] is the i'th observation or RV

    """
    print >> sys.stderr, switch_errmsg

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



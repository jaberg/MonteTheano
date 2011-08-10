"""
"""

import numpy

import theano
from theano.compile import rebuild_collect_shared
from theano import tensor
from theano.gof.graph import ancestors

from .pdfreg import pdf


###########################################################
# Stock proposal distributions
# for random variables in theano.tensor.raw_random
###########################################################


def full_sample(s_rng, outputs ):
    """
    Return dictionary mapping random vars to their symbolic samples.
    """
    all_vars = ancestors(outputs)
    assert outputs[0] in all_vars

    RVs = [v for v in all_vars if is_random_var(v)]

    rdict = dict([(v, v) for v in RVs])
        # outputs is same
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


def mh_sample(s_rng, outputs, observations):
    # TODO: should there be a size variable here?
    # TODO: implement lag and burn-in
    # TODO: implement size
    """
    Return a dictionary mapping random variables to their sample values.
    """

    if observations is None:
        raise NotImplementedError()

    all_vars = ancestors(list(outputs) + list(observations.keys()))

    for o in observations:
        assert o in all_vars
        if not is_random_var(o):
            raise TypeError(o)

    RVs = [v for v in all_vars if is_random_var(v)]

    free_rvs = [v for v in RVs if v not in observations]

    # sample from the prior to initialize these guys?
    free_rvs_state = [theano.shared(numpy.array(size=(1,1,1)))
            for v in free_rvs]

    current_likelihood = theano.shared(numpy.array(float('-inf')))

    proposals = [mh_proposal(rv) for rv in free_rvs]

    full_observations = dict(observations)
    full_observations.update(
            dict([(rv, s) for rv, s in zip(free_rvs, proposals)]))

    new_likelihood = likelihood(observations=full_observations)

    accept = fn(new_likelihood, current_likelihood)

    new_states = [tensor.switch(accept, proposed, current)
            for (proposed, current) in zip(proposals, free_rvs_state)]

    updates = zip(free_rvs_state, new_states)

    # install updates as default_updates like the seeds of normal random
    # numbers, and then return the dictionary mapping random variables to their
    # shared var states to the caller.


    #TODO: think - if a proposal is rejected, is is the current state to be used
    # as multiple consecutive samples?



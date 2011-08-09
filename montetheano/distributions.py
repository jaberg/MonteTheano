"""
"""

import numpy

from theano import tensor
from theano.gof.graph import stack_search, deque

class Normal(object):

    def pdf(self, node, sample):
        random_state, size, avg, std = node.inputs

        # assume sample has sample.shape == size

        Z = (2 * pi * std**2)
        E = 0.5 * ((avg - sample)/std)**2

        return tensor.prod(tensor.exp(-E) / Z)

def is_random_var(v):
    #TODO: How to make this work with non-standard RandomStreams?

    if v.owner and isinstance(v.owner.op, tensor.raw_random.RandomFunction):
        return True
    return False

def ancestors(variable_list, blockers = None):
    """Return the inputs required to compute the given Variables.

    :type variable_list: list of `Variable` instances
    :param variable_list:
        output `Variable` instances from which to search backward through owners
    :rtype: list of `Variable` instances
    :returns:
        input nodes with no owner, in the order found by a left-recursive depth-first search
        started at the nodes in `variable_list`.

    """
    def expand(r):
        if r.owner and (not blockers or r not in blockers):
            l = list(r.owner.inputs)
            l.reverse()
            return l
    dfs_variables = stack_search(deque(variable_list), expand, 'dfs')
    return dfs_variables

def clone_with_givens(var, givens):
    return theano.compile.pfunc.rebuild_collect_shared(
            [var],
            replace=givens)



def likelihood(observations):
    """
    \sum_i log(P(observations)) given that observations[i] ~ RV, iid.

    observations: a dictionary mapping random variables to tensors.

        observations[RV] = rv_observations

        rv_observations[i] is the i'th observation or RV

    """
    rval = []
    for rv in observations:
        rv_lik = rv.owner.op.likelihood(observations[rv])
        cloned_rv_lik = clone_with_givens(rv_lik, givens=observations)
        rval.append(cloned_rv_lik)

    return tensor.prod(*rval)

def sample(s_rng, outputs, size):
    """
    Return dictionary mapping random vars to their symbolic samples.
    """
    all_vars = ancestors(outputs)
    assert outputs[0] in all_vars

    RVs = [v for v in all_vars if is_random_var(v)]

    if size == () or size == []:
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



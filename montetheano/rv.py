"""
Functions for operating on random variables.
"""
import theano
from theano import tensor
from for_theano import ancestors, as_variable, clone_keep_replacements
import rstreams


def is_randomstate(var):
    """
    """
    return isinstance(var.type, rstreams.randomstate_types)


def is_rv(var, blockers=None):
    """
    Return True iff var is a random variable.

    A random variable is a variable with a randomstate object in its ancestors.
    """    
    #TODO: could optimize by stopping the recusion as soon as a randomstate is
    #      found
    return any(is_randomstate(v) for v in ancestors([var], blockers=blockers))


def is_raw_rv(var):
    """
    Return True iff v is the result of calling s_rng.something()
    """
    return var.owner and is_randomstate(var.owner.inputs[0])


def all_raw_rvs(outputs):
    """
    Return a list of all random variables required to compute `outputs`.
    """
    all_vars = ancestors(outputs)
    assert outputs[0] in all_vars
    rval = [v for v in all_vars if is_raw_rv(v)]
    return rval


def typed_items(dct):
    return dict([
        (rv, as_variable(sample, type=rv.type))
        for rv, sample in dct.items()])


def condition(rvs, observations):
    if len(rvs) > 1:
        raise NotImplementedError()
    observations = typed_items(observations)
    print observations
    # if none of the rvs show up in the ancestors of any observations
    # then this is easy conditioning
    obs_ancestors = ancestors(observations.keys(), blockers=rvs)
    if any(rv in obs_ancestors for rv in rvs):
        # not-so-easy conditioning
        # we need to produce a sampler-driven model
        raise NotImplementedError()
    else:
        # easy conditioning
        rvs_anc = ancestors(rvs, blockers=observations.keys())
        frontier = [r for r in rvs_anc
                if r.owner is None or r in observations.keys()]
        cloned_inputs, cloned_outputs = clone_keep_replacements(frontier, rvs,
                replacements=observations)
        return cloned_outputs

# TODO: does this function belong here or in rstreams
def lpdf(rv, sample, **kwargs):
    """
    Return the probability (density) that random variable `rv`, returned by
    a call to one of the sampling routines of this class takes value `sample`
    """
    if not is_rv(rv):
        raise TypeError('rv not recognized as a random variable', rv)

    if is_raw_rv(rv):
        dist_name = rstreams.rv_dist_name(rv)
        pdf = rstreams.pdfs[dist_name]
        return pdf(rv.owner, sample, kwargs)
    else:
        #TODO: infer from the ancestors of v what distribution it
        #      has.
        raise NotImplementedError()

def conditional_log_likelihood(assignment, givens):
    """
    Return log(P(rv0=sample | given))

    assignment: rv0=val0, rv1=val1, ...
    given: var0=v0, var1=v1, ...

    Each of val0, val1, ... v0, v1, ... is supposed to represent an identical
    number of draws from a distribution.  This function returns the real-valued
    density for each one of those draws.

    The output from this function may be a random variable, if not all sources
    of randomness are removed by the assignment and the given.
    """
    
    for rv in assignment.keys():
        if not is_rv(rv):
            raise ValueError('non-random var in assignment key', rv)
    
    # Cast assignment elements to the right kind of thing
    assignment = typed_items(assignment)
    
    rvs = assignment.keys()
    #TODO: this is not ok for undirected models
    #      we need to be able to let condition introduce joint
    #      dependencies somehow.
    #      The trouble is that lpdf wants to get the pdfs one variable at a
    #      time.  That makes sense for directed models, but not for
    #      undirected ones.
    new_rvs = condition(rvs, givens)
    return full_log_likelihood(
            [(new_rv, assignment[rv])
                for (new_rv, rv) in zip(new_rvs, rvs)],
            given={})

def full_log_likelihood(assignment):
    """
    Return log(P(rv0=sample))

    assignment: rv0=val0, rv1=val1, ...

    Each of val0, val1, ... v0, v1, ... is supposed to represent an identical
    number of draws from a distribution.  This function returns the real-valued
    density for each one of those draws.

    The output from this function may be a random variable, if not all sources
    of randomness are removed by the assignment and the given.
    """

    for rv in assignment.keys():
        if not is_rv(rv):
            raise ValueError('non-random var in assignment key', rv)

    # All random variables that are not assigned should stay as the same object so it can later be replaced
    # If this is not done this way, they get cloned
    RVs = [v for v in ancestors(assignment.keys()) if is_raw_rv(v)]
    for rv in RVs:
        if rv not in assignment:
            assignment[rv] = rv
                
    # Cast assignment elements to the right kind of thing
    assignment = typed_items(assignment)

    pdfs = [lpdf(rv, sample) for rv, sample in assignment.items()]
    lik = tensor.add(*[tensor.sum(p) for p in pdfs])
    
    dfs_variables = ancestors([lik], blockers=assignment.keys())
    frontier = [r for r in dfs_variables
            if r.owner is None or r in assignment.keys()]
    cloned_inputs, cloned_outputs = clone_keep_replacements(frontier, [lik],
            replacements=assignment)
    cloned_lik, = cloned_outputs
    return cloned_lik


def energy(assignment, given):
    """
    Return -log(P(rv0=sample | given)) +- const

    assignment: rv0=val0, rv1=val1, ...
    given: var0=v0, var1=v1, ...

    Each of val0, val1, ... v0, v1, ... is supposed to represent an identical
    number of draws from a distribution.  This function returns the real-valued
    density for each one of those draws.

    The output from this function may be a random variable, if not all sources
    of randomness are removed by the assignment and the given.
    """
    try:
        return -conditional_log_likelihood(assignment, given)
    except:
        # get the log_density up to an additive constant
        raise NotImplementedError()

"""
Functions for operating on random variables.
"""
from for_theano import ancestors
from rstreams import randomstate_types, lpdf
from shallow_clone import clone_keep_replacements


def is_randomstate(var):
    """
    """
    return isinstance(var.type, randomstate_types)


def is_rv(var, blockers=None):
    """
    Return True iff var is a random variable.

    A random variable is a variable with a randomstate object in its ancestors.
    """
    #TODO: could optimize by stopping the recusion as soon as a randomstate is
    #      found
    return any(is_randomstate(v) for v in ancestors(var, blockers=blockers))


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

def condition(rvs, observations):
    raise NotImplementedError()

def log_density(assignment, given):
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
    assignment = dict([
        (rv, rv.type.filter(sample, allow_downcast=True))
        for rv, sample in assignment.items()])

    if givens:
        rvs = assignment.keys()
        #TODO: this is not ok for undirected models
        #      we need to be able to let condition introduce joint
        #      dependencies somehow.
        #      The trouble is that lpdf wants to get the pdfs one variable at a
        #      time.  That makes sense for directed models, but not for
        #      undirected ones.
        new_rvs = condition(rvs, givens)
        return log_density(
                [(new_rv, assignment[rv])
                    for (new_rv, rv) in zip(new_rvs, rvs)],
                given={})
    else:
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
        return -log_density(assignment, given)
    except:
        # get the log_density up to an additive constant
        raise NotImplementedError()


def full_log_likelihood(observations):
    """
    \sum_i log(P(observations)) given that observations[i] ~ RV, iid.

    observations: a dictionary mapping random variables to tensors.

        observations[RV] = rv_observations

        rv_observations[i] is the i'th observation or RV

    """
    rval = log_density(observations, {})

    if is_rv(rval):
        raise ValueError('insufficient information for full_likelihood')

    return rval


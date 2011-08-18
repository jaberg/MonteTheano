from for_theano import ancestors
from rstreams import randomstate_types
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


def density(assignment, observations):
    """
    Return P(rv0=sample | observations)

    assignment: rv0=val0, rv1=val1, ...
    observations: var0=v0, var1=v1, ...

    Each of val0, val1, ... v0, v1, ... is supposed to represent an identical
    number of draws from a distribution.  This function returns the real-valued
    density for each one of those draws.

    The output from this function may be a random variable, if not all sources
    of randomness are removed by the assignment and the observations.
    """
    raise NotImplementedError()


def energy(assignment, observations):
    """
    Return -log(P(rv0=sample | observations)) +- const

    assignment: rv0=val0, rv1=val1, ...
    observations: var0=v0, var1=v1, ...

    Each of val0, val1, ... v0, v1, ... is supposed to represent an identical
    number of draws from a distribution.  This function returns the real-valued
    density for each one of those draws.

    The output from this function may be a random variable, if not all sources
    of randomness are removed by the assignment and the observations.
    """
    raise NotImplementedError()


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
    observations = dict([(rv, tensor.as_tensor_variable(obs).astype(rv.dtype))
        for rv, obs in observations.items()])

    pdfs = [log_pdf(rv, obs) for rv,obs in observations.items()]

    lik = tensor.add(*[tensor.sum(p) for p in pdfs])

    dfs_variables = ancestors([lik], blockers=RVs)
    frontier = [r for r in dfs_variables if r.owner is None or r in RVs]

    # Benjamin - why not use observations here?
    cloned_inputs, cloned_outputs = clone_keep_replacements(frontier, [lik],
            replacements=dict(observations.items()))

    cloned_lik, = cloned_outputs
    return cloned_lik


"""
Algorithms for drawing samples by MCMC

"""


def full_sample(s_rng, outputs ):
    all_vars = ancestors(outputs)
    assert outputs[0] in all_vars
    RVs = [v for v in all_vars if is_random_var(v)]
    rdict = dict([(v, v) for v in RVs])

    if True:
        # outputs is same
        raise NotImplementedError()
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

# Sample the generative model and return "outputs" for cases where "condition" is met.
# If no condition is given, it just samples from the model
# The outputs can be a single TheanoVariable or a list of TheanoVariables.
# The function returns a single sample or a list of samples, depending on "outputs"; and an updates dictionary.
def rejection_sample(outputs, condition = None):
    if isinstance(outputs, tensor.TensorVariable):
        init = [0]
    else:
        init = [0]*len(outputs)
    if condition is None:
        # TODO: I am just calling scan to get updates, can't I create this myself?
        # output desired RVs when condition is met
        def rejection():
            return outputs

        samples, updates = theano.scan(rejection, outputs_info = init, n_steps = 1)
    else:
        # output desired RVs when condition is met
        def rejection():
            return outputs, {}, theano.scan_module.until(condition)
        samples, updates = theano.scan(rejection, outputs_info = init, n_steps = 1000)
    if isinstance(samples, tensor.TensorVariable):
        sample = samples[-1]
    else:
        sample = [s[-1] for s in samples]
    return sample, updates

def mh_sample(s_rng, outputs, observations = {}):
    # TODO: should there be a size variable here?
    # TODO: implement lag and burn-in
    # TODO: implement size
    """
    Return a dictionary mapping random variables to their sample values.
    """

    all_vars = ancestors(list(outputs) + list(observations.keys()))
    for o in observations:
        assert o in all_vars
        if not is_random_var(o):
            raise TypeError(o)

    free_RVs = [v for v in RVs if v not in observations]

    # TODO: sample from the prior to initialize these guys?
    # free_RVs_state = [theano.shared(v) for v in free_RVs]
    # TODO: how do we infer shape?
    free_RVs_state = [theano.shared(0.5*numpy.ones(shape=())) for v in free_RVs]
    free_RVs_prop = [s_rng.normal(size = (), std = .1) for v in free_RVs]

    log_likelihood = theano.shared(numpy.array(float('-inf')))

    U = s_rng.uniform(size=(), low=0, high=1.0)

    # TODO: can we pre-generate the noise
    def mcmc(ll, *frvs):
        # TODO: implement generic proposal distributions
        # TODO: how do we infer shape?
        proposals = [(rvs + rvp) for rvs,rvp in zip(free_RVs_state, free_RVs_prop)]

        full_observations = dict(observations)
        full_observations.update(dict([(rv, s) for rv, s in zip(free_RVs, proposals)]))

        new_log_likelihood = full_log_likelihood(observations = full_observations)

        accept = tensor.or_(new_log_likelihood > ll, U <= tensor.exp(new_log_likelihood - ll))

        return [tensor.switch(accept, new_log_likelihood, ll)] + \
               [tensor.switch(accept, p, f) for p, f in zip(proposals, frvs)], \
               {}, theano.scan_module.until(accept)

    samples, updates = theano.scan(mcmc, outputs_info = [log_likelihood] + free_RVs_state, n_steps = 10000000)
    updates[log_likelihood] = samples[0][-1]
    updates.update(dict([(f, s[-1]) for f, s in zip(free_RVs_state, samples[1:])]))
    
    return [free_RVs_state[free_RVs.index(out)] for out in outputs], log_likelihood, updates

def hybridmc_sample(s_rng, outputs, observations = {}):
    # TODO: should there be a size variable here?
    # TODO: implement lag and burn-in
    # TODO: implement size
    """
    Return a dictionary mapping random variables to their sample values.
    """

    all_vars = ancestors(list(outputs) + list(observations.keys()))
    
    for o in observations:
        assert o in all_vars
        if not is_random_var(o):
            raise TypeError(o)

    RVs = [v for v in all_vars if is_random_var(v)]

    free_RVs = [v for v in RVs if v not in observations]
    
    free_RVs_state = [theano.shared(0.5*numpy.ones(shape=())) for v in free_RVs]    
    free_RVs_prop = [s_rng.normal(size = (), std = 1) for v in free_RVs]    
    
    log_likelihood = theano.shared(numpy.array(float('-inf')))
    
    U = s_rng.uniform(size=(), low=0, high=1.0)
    
    epsilon = numpy.sqrt(2*0.03)
    def mcmc(ll, *frvs):
        full_observations = dict(observations)
        full_observations.update(dict([(rv, s) for rv, s in zip(free_RVs, frvs)]))
        loglik = -full_log_likelihood(observations = full_observations)

        proposals = free_RVs_prop
        H = tensor.add(*[tensor.sum(tensor.sqr(p)) for p in proposals])/2. + loglik

# -- this should be an inner loop
        g = tensor.grad(loglik, frvs)
        proposals = [(p - epsilon*g/2.) for p, g in zip(proposals, g)]

        rvsp = [(rvs + epsilon*rvp) for rvs,rvp in zip(frvs, proposals)]
        
        full_observations = dict(observations)
        full_observations.update(dict([(rv, s) for rv, s in zip(free_RVs, rvsp)]))
        new_loglik = -full_log_likelihood(observations = full_observations)
        
        gnew = tensor.grad(new_loglik, rvsp)
        proposals = [(p - epsilon*gn/2.) for p, gn in zip(proposals, gnew)]
# --
        
        Hnew = tensor.add(*[tensor.sum(tensor.sqr(p)) for p in proposals])/2. + new_loglik

        dH = Hnew - H
        accept = tensor.or_(dH < 0., U < tensor.exp(-dH))

        return [tensor.switch(accept, -new_loglik, ll)] + \
            [tensor.switch(accept, p, f) for p, f in zip(rvsp, frvs)], \
            {}, theano.scan_module.until(accept)

    samples, updates = theano.scan(mcmc, outputs_info = [log_likelihood] + free_RVs_state, n_steps = 10000000)
    
    updates[log_likelihood] = samples[0][-1]
    updates.update(dict([(f, s[-1]) for f, s in zip(free_RVs_state, samples[1:])]))
    
    return [free_RVs_state[free_RVs.index(out)] for out in outputs], log_likelihood, updates


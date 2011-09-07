"""
Algorithms for drawing samples by MCMC

"""
import numpy
import theano
from theano import tensor
from for_theano import ancestors, infer_shape, evaluate_with_assignments, evaluate
from rv import is_raw_rv, full_log_likelihood, lpdf


# Major TODOs:
# - RVs should have a non-symbolic shape so the MC states can be allocated
# - We need to initialize the chains in draw from the independent prior distributions
# - We need proposal distributions for all RVs from which to draw samples
# - An additional loop around mh_sample is required
# - An efficient parallel MC sampler is possible, which might be less decorrelated (or more book-keeping is required)
# - The HMC sampler needs an outside loop and an additional inner loop for the leap-frog steps



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
    all_vars = ancestors(list(outputs) + list(observations.keys()))
    
    for o in observations:
        assert o in all_vars
        if not is_raw_rv(o):
            raise TypeError(o)

    RVs = [v for v in all_vars if is_raw_rv(v)]
    free_RVs = [v for v in RVs if v not in observations]

    # Draw sample from the proposal
    free_RVs_state = []
    for v in free_RVs:
        f = theano.function([], v,
                mode=theano.Mode(linker='py', optimizer=None))
        free_RVs_state.append(theano.shared(f()))

    log_likelihood = theano.shared(numpy.array(float('-inf')))

    U = s_rng.uniform(low=0.0, high=1.0)

    def mcmc(ll, *frvs):
        proposals = [s_rng.local_proposal(v, rvs) for v, rvs in zip(free_RVs, frvs)]
        proposals_rev = [s_rng.local_proposal(v, rvs) for v, rvs in zip(free_RVs, proposals)]

        full_observations = dict(observations)
        full_observations.update(dict([(rv, s) for rv, s in zip(free_RVs, proposals)]))
        new_log_likelihood = full_log_likelihood(full_observations)

        logratio = new_log_likelihood - ll \
            + tensor.add(*[tensor.sum(lpdf(p, r)) for p, r in zip(proposals_rev, frvs)]) \
            - tensor.add(*[tensor.sum(lpdf(p, r)) for p, r in zip(proposals, proposals)])
                   
        accept = tensor.gt(logratio, tensor.log(U))
        
        return [tensor.switch(accept, new_log_likelihood, ll)] + \
               [tensor.switch(accept, p, f) for p, f in zip(proposals, frvs)], \
               {}, theano.scan_module.until(accept)

    samples, updates = theano.scan(mcmc, outputs_info = [log_likelihood] + free_RVs_state, n_steps = 100)
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
        if not is_raw_rv(o):
            raise TypeError(o)

    RVs = [v for v in all_vars if is_raw_rv(v)]

    free_RVs = [v for v in RVs if v not in observations]
    
    free_RVs_state = [theano.shared(numpy.ones(shape=infer_shape(v)), broadcastable=tuple(numpy.asarray(infer_shape(v))==1)) for v in free_RVs]
    free_RVs_prop = [s_rng.normal(0, 1, draw_shape=infer_shape(v)) for v in free_RVs]
    
    log_likelihood = theano.shared(numpy.array(float('-inf')))
    
    U = s_rng.uniform(low=0, high=1.0)
    
    epsilon = numpy.sqrt(2*0.03)
    def mcmc(ll, *frvs):
        full_observations = dict(observations)
        full_observations.update(dict([(rv, s) for rv, s in zip(free_RVs, frvs)]))
        
        loglik = -full_log_likelihood(full_observations)

        proposals = free_RVs_prop
        H = tensor.add(*[tensor.sum(tensor.sqr(p)) for p in proposals])/2. + loglik

# -- this should be an inner loop
        g = []
        g.append(tensor.grad(loglik, frvs))
        
        proposals = [(p - epsilon*g/2.) for p, g in zip(proposals, g)]

        rvsp = [(rvs + epsilon*rvp) for rvs,rvp in zip(frvs, proposals)]
        
        full_observations = dict(observations)
        full_observations.update(dict([(rv, s) for rv, s in zip(free_RVs, rvsp)]))
        new_loglik = -full_log_likelihood(full_observations)
        
        gnew = []
        gnew.append(tensor.grad(new_loglik, rvsp))
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

def mh2_sample(s_rng, outputs, observations = {}):    
    all_vars = ancestors(list(observations.keys()) + list(outputs))
        
    for o in observations:
        assert o in all_vars
        if not is_raw_rv(o):
            raise TypeError(o)
    
    RVs = [v for v in all_vars if is_raw_rv(v)]
    free_RVs = [v for v in RVs if v not in observations]
    
    free_RVs_state = []
    for v in free_RVs:
        f = theano.function([], v,
                mode=theano.Mode(linker='py', optimizer=None))
        free_RVs_state.append(theano.shared(f()))
    
    U = s_rng.uniform(low=0.0, high=1.0)
    
    rr = []
    for index in range(len(free_RVs)):
        # TODO: why does the compiler crash when we try to expose the likelihood ?
        full_observations = dict(observations)
        full_observations.update(dict([(rv, s) for rv, s in zip(free_RVs, free_RVs_state)]))
        log_likelihood = full_log_likelihood(full_observations)
        
        proposal = s_rng.local_proposal(free_RVs[index], free_RVs_state[index])
        proposal_rev = s_rng.local_proposal(free_RVs[index], proposal)

        full_observations = dict(observations)
        full_observations.update(dict([(rv, s) for rv, s in zip(free_RVs, free_RVs_state)]))
        full_observations.update(dict([(free_RVs[index], proposal)]))
        new_log_likelihood = full_log_likelihood(full_observations)

        bw = tensor.sum(lpdf(proposal_rev, free_RVs_state[index]))
        fw = tensor.sum(lpdf(proposal, proposal))

        lr = new_log_likelihood-log_likelihood+bw-fw

        accept = tensor.gt(lr, tensor.log(U))

        updates = {free_RVs_state[index] : tensor.switch(accept, proposal, free_RVs_state[index])}
        rr.append(theano.function([], [accept], updates=updates))
    
    # TODO: this exacte amount of samples given back is still wrong
    def sampler(nr_samples, burnin = 100, lag = 100):
        data = [[] for o in outputs]
        for i in range(nr_samples*lag+burnin):        
            accept = False
            while not accept:
                index = numpy.random.randint(len(free_RVs))

                accept = rr[index]()            
                if accept and i > burnin and (i-burnin) % lag == 0:
                    for d, o in zip(data, outputs):
                        # TODO: this can be optimized
                        if is_raw_rv(o):
                            d.append(free_RVs_state[free_RVs.index(o)].get_value())
                        else:
                            full_observations = dict(observations)
                            full_observations.update(dict([(rv, s) for rv, s in zip(free_RVs, free_RVs_state)]))
                            d.append(evaluate(evaluate_with_assignments(o, full_observations)))
        data = [numpy.asarray(d).squeeze() for d in data]
        
        return data
    
    return sampler
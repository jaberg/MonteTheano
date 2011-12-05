import numpy
import theano
from theano import tensor
from for_theano import ancestors, infer_shape, evaluate_with_assignments, evaluate
from rv import is_raw_rv, full_log_likelihood, lpdf

def likelihood_gradient(observations = {}, learning_rate = 0.1):
    all_vars = ancestors(list(observations.keys()))
    
    for o in observations:
        assert o in all_vars
        if not is_raw_rv(o):
            raise TypeError(o)

    RVs = [v for v in all_vars if is_raw_rv(v)]
    free_RVs = [v for v in RVs if v not in observations]

    # Instantiate actual values for the different random variables:
    params = dict()
    for v in free_RVs:
        f = theano.function([], v, mode=theano.Mode(linker='py', optimizer=None))
        params[v] = theano.shared(f())

    # Compute the full log likelihood:
    full_observations = dict(observations)
    full_observations.update(params)    
    log_likelihood = full_log_likelihood(full_observations)
    
    # Construct the update equations for learning:
    updates = dict()
    for frvs in params.values():
        updates[frvs] = frvs + learning_rate * tensor.grad(log_likelihood, frvs)
        
    return params, updates, log_likelihood




class Model(object):
    """
    Model is a set of random variables.

    Many Random Variables can share the same parameters.
    For example n = Normal() and 2 * n.

    """
    def __init__(self):
        self.rv_set = set([])

    def add(self, rv):
        self.rvs.add(rv)

    def maximum_likelihood(observations):
        """
        """


class RV(object):
    """
    A Random Variable is a probability distribution over multiple possible values.

    It is similar in some senses to a Theano Type, in that a Type identifies a set of possible
    values, whereas a Random Variable comes with a measure over that set of values.

    Any non-RV is treated for some intents and purposes
    as a Random Variable with a dirac density. This justifies sample(3) -> 3 for example,
    because 3 is taken to represent a dirac density over the 3 object.

    This RV corresponds to a node in a directed graphical model.
    """

    def sample(self, N):
        """
        """
        raise NotImplementedError('implement me')

    def params(self):
        raise NotImplementedError('implement me')
        return []

    def pdf_given_parents(self, x, param_exprs):
        """
        Return the probability / probabilities of x, conditioned on given values for parents.
        """
        raise NotImplementedError('implement me')

    def mode_given_parents(self):
        """
        Return Theano expression for most probable sample.
        """
        raise NotImplementedError('implement me')


def sample_memo(obj, draw_shape, rstreams, memo):
    try:
        return memo[id(obj)]
    except KeyError:
        pass
    def rec_sample(thing, shp):
        return sample_memo(thing, shp, rstreams, memo)
    rval = obj.sample(draw_shape, rstreams, rec_sample)
    memo[id(obj)] = rval
    return rval


def sample(objects, draw_shape, rstreams):
    """
    Sample N values from distribution `obj`.

    If obj is not a random variable, then it is returned unchanged.
    """
    memo = {}
    for o in objects:
        sample_memo(o, draw_shape, rstreams, {})
    return [memo[o] for o in objects]


def as_rv_or_sharedX(thing):
    if isinstance(thing, theano.Variable):
        return thing
    if isinstance(thing, RV):
        return thing
    return tensor.shared(numpy.asarray(thing, dtype=theano.config.floatX))


class Dirac(RV):
    """
    Degenerate distribution - takes only one possible value.
    """
    def __init__(self, val):
        if not isinstance(val, theano.Variable):
            raise TypeError()
        self.val = val

    def sample(self, draw_shape, rstreams, rec_sample):
        if draw_shape:
            raise NotImplementedError()
        else:
            return self.val

    def mode(self):
        return self.val


class Normal(RV):
    def __init__(self, mu, sigma):
        self.mu = as_rv_or_sharedX(mu)
        self.sigma = as_rv_or_sharedX(sigma)

    def sample(self, draw_shape, rstreams, sample):
        mu = sample(self.mu, ())
        sigma = sample(self.sigma, ())
        return rstreams.normal(draw_shape, mu, sigma)

    def pdf(self, x):
        one = tensor.as_tensor_variable(numpy.asarray(1, dtype='float32'))
        rstate, shape, mu, sigma = node.inputs
        Z = tensor.sqrt(2 * numpy.pi * sigma**2)
        E = 0.5 * ((mu - x)/(one*sigma))**2
        return tensor.exp(-E) / Z

    def params(self):
        return [i for i in [self.mu, self.sigma]
                if isinstance(i, theano.Variable)]

    def posterior(self, x, weights=None):
        """
        Message-passing required.
        """
        raise NotImplementedError() 

    def maximum_likelihood(self, x, weights=None):
        eps = 1e-8
        if weights is None:
            new_mu = tensor.mean(sample)
            new_sigma = tensor.std(sample)

        else:
            denom = tensor.maximum(tensor.sum(weights), eps)
            new_mu = tensor.sum(sample*weights) / denom
            new_sigma = tensor.sqrt(
                    tensor.sum(weights * (sample - new_mu)**2)
                    / denom)
        rval = Updates()
        if self.mu in self.params():
            rval[self.mu] = new_mu
        if self.sigma in self.params():
            rval[self.sigma] = new_sigma
        return rval


class Categorical(RV):
    def __init__(self, weights):
        self.weights = weights

    def sample(self, N, rstreams):
        raise NotImplementedError()

    def pdf(self, x):
        raise NotImplementedError()


class List(RV):
    def __init__(self, components):
        self.components = components

    def __getitem__(self, idx):
        if isinstance(idx, Categorical):
            return Mixture(Categorical.weights, self.components)
        else:
            return self.components[idx]

class Dict(RV):
    def __init__(self, **kwargs):
        self.components = kwargs

    def sample(self, draw_shape, rstreams, memo):
        raise NotImplementedError()

    def pdf(self, X):

class Mixture(RV):
    def __init__(self, weights, components):
        self.weights = weights
        self.components = components

    def sample(self, draw_shape, rstreams, memo):
        raise NotImplementedError()

    def pdf(self, x):
        raise NotImplementedError()


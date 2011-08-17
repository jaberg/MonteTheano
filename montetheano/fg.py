
class Updates(dict):
    def __add__(self, other):
        rval = Updates(self)
        rval += other  # see: __iadd__
        return rval
    def __iadd__(self, other):
        d = dict(other)
        for k,v in d.items():
            if k in self:
                raise KeyError()

            self[k] = v


class FG_node(object):
    def neibs_except(self, skip):
        return [n for n in self.neibs if n is not skip]

    def message_to(self, dest, messages):
        raise NotImplementedError()


class FG_message(object):
    """
    """
    def scale_by(self, other):
        raise NotImplementedError()

    def normalize(self):
        pass

    def sample(self):
        raise NotImplementedError()

    def mode(self):
        raise NotImplementedError()

    def support_min(self):
        raise NotImplementedError()

    def support_max(self):
        raise NotImplementedError()

    def mean(self):
        raise NotImplementedError()

    def var(self):
        raise NotImplementedError()


class FG_1(FG_message):
    def scale_by(self, other):
        return other


class Table(FG_message):
    def __init__(self, weight):
        self.weight = weight

    def scale_by(self, other):
        if isinstance(other, FG_1):
            return self
        self.weight = self.weight * other.weight

    def normalize(self):
        self.weight = self.weight / (self.weight.sum() + 1e-8)


class Normal(FG_message, rv.Normal):

    def scale_by(self, other):
        if isinstance(other, FG_1):
            return self
        if isinstance(other, Particles):
            return other.scale_by(self)
        if isinstance(other, Normal):
            raise NotImplementedError()
        raise TypeError(other)


class Particles(FG_message):
    def __init__(self, particles, weights):
        self.particles = particles
        self.weights = weights

    def scale_by(self, other):
        if isinstance(other, FG_1):
            return self
        if isinstance(other, Particles):
            raise NotImplementedError()
        if isinstance(other, Normal):
            return Particles(
                    particles,
                    weights * other.pdf(particles))
        raise TypeError(other)


class FG_var(FG_node):
    """
    """
    def __init__(self, neibs):
        self.neibs = neibs

    def message_to(self, dest, messages):
        """
        Return the product of messages from not-dest to self
        """
        rval = FG_1()
        for n in self.neibs_except(dest):
            rval.scale_by(messages[(n, self)])
        return rval


class FG_fun(FG_node):
    """
    """
    def message_to(self, dest, messages):
        """
        Return the message corresponding to the input node not appearing in the messages

        messages: dict FG_var -> FG_message
        """

class FG_NormalPotential(FG_fun):
    def __init__(self, mu, sigma, x):
        self.mu = mu
        self.sigma = sigma
        self.x = x
        self.neibs = [mu, sigma, x]

    def message_to(self, dest, messages):
        # f(mu, sigma, x) g(mu), h(sigma) i(x)
        m_mu = messages[(self.mu, self)]
        m_sigma = messages[(self.sigma, self)]
        m_x = messages[(self.x, self)]
        if dest is mu:
            # \sum_{sigma, x} f(mu, sigma, x) h(sigma) i(x)
            if isinstance(m_sigma, FG_1):
                # \sum_{sigma, x} f(mu, sigma, x) i(x)
                # \sum_{sigma, x} exp(-.5 (\mu-x)^2/\sigma^2)/(\sqrt{2 pi} \sigma)
                #                 i(x)
                if isinstance(m_x, Particles):
                    # i(x) = \sum_i w_i \dirac(x_i)
                    #
                    # \sum_{sigma, x} f(mu, sigma, x) i(x)
                    # \sum_{sigma, i} w_i exp(-.5 (\mu-x_i)^2/\sigma^2)/(\sqrt{2 pi} \sigma)
                    #                 w_i
            else:
                raise NotImplementedError()




def bp(edges, schedule):
    for (i,j) in schedule:
        # message from i to j
        message[(i,j)]

        for outgoing in edges(n):

        # outgoing from 
def posteriors(rvs, observations):
    """
    rvs: a list of random variables

    observations: a dictionary whose keys are random variables and whose values are
                  obeservations of those random variables.

    returns: random variables for each rv in rvs representing posterior distributions
    """
    all_rvs = ancestors(list(rvs) + list(observations.keys()))

    # initialize messages
    old_messages = {}
    for i in all_rvs:
        for j in i.params() + i.clients():
            old_messages[(i,j)] = j.init_message(i)

    # according to some message update schedule...
    new_messages = {}
    for i in schedule(all_rvs, observations):
        for j in i.params():
            # send message from i to j
        for j in i.params() + i.clients():
            h_i = 1
            for k in i.params() + i.clients():
                if k is not j:
                    h_i = old_messages[(k,i)].scale_by(h_i)

            # sum over possible values for rv_i
            #    product of 
            new_messages[(i, j)] = 
            old_messages[(i,j)] = i.init_message()

    raise NotImplementedError()

def posterior_modes(observations, algo='bp'):
    """
    observations: a dictionary whose keys are random variables and whose values are
                  obeservations of those random variables.

    returns: Updates for the maximum likelihood estimates for all shared variables affecting the
        likelihood of these observations.
    """
    if algo

    # This method works by max-product belief propagation


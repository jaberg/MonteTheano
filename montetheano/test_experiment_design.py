import unittest
import numpy
import theano
from theano import tensor
from for_theano import where
from pdfreg import RVs

class TestSimple(unittest.TestCase):

    def setUp(self):

        s_rng = tensor.shared_randomstreams.RandomStreams(23424)

        n_draws = tensor.lscalar()
        v_draws = n_draws.dimshuffle('x')
        spec = self.spec = {}
        spec['lr'] = 10**s_rng.uniform(size=v_draws, low=-5, high=0, ndim=1)
        spec['pp'] = s_rng.uniform(size=v_draws, ndim=1) < 0.5

        spec_items = spec.items()
        rvs = self.rvs = RVs(spec.values()) # symbolic uniform draws
        assert len(rvs) == 2

        # we want to minimize this
        # which is normally not something we have as an expression
        y = (s_rng.normal(size=v_draws, ndim=1)*0.1
                - spec['pp']
                + rvs[0]**2
                + rvs[1]**2)

        spec_values = [v for (k,v) in spec_items] # symbolic lr, pp

        self.draw_from_prior = theano.function([n_draws], [y] + spec_values + rvs)

        draws = self.draw_from_prior(10)
        self.yvals = draws[0]

        self.observations = dict(zip([k for (k,v) in spec_items], draws[1:]))
        self.random_samples = dict(zip(rvs, draws[1+len(spec_items):]))


    def test_draw_from_prior(self):
        print self.draw_from_prior(5)

    def _algo1(self):

        algo1 = Algo1(self.spec)
        for i in xrange(100):
            new_point = algo1.suggest_trial(random_samples, losses)


        good_model = model_from_spec(self.spec, size=len(good_data))
        bad_model = model_from_spec(self.spec, size=len(bad_data))

        good_model.fit(good_data)
        bad_model.fit(bad_data)

        proposal = good_data
        best_sample = maximize(
                tensor.log(
                    likelihood(good_model, proposal)
                    / likelihood(bad_model, proposal)))

        # design decisions
        mu_llr_good, sigma_llr_good = AdaptiveParzen()(data_llr_good, low=-5,
                high=-1.5, minsigma=0.01)
        mu_llr_bad, sigma_llr_bad  = AdaptiveParzen()(data_llr_bad,  low=-5,
                high=-1.5, minsigma=0.01)


        rv_llr_good = gauss_mixture(s_rng, mu=mu_llr_good, sigma=sigma_llr_good)
        self.sample_llr = s_rng.normal(mean=-4, std=2, size=(5,))

        self.sample_llr_logprob = log_density(self.sample_llr, rv_llr_good)

        if 0:

            rv_llr_bad =  gauss_mixture(s_rng, mu=mu_llr_bad,  sigma=sigma_llr_bad)

            self.rv_llr = tensor.switch(rv_err_good, rv_llr_good, rv_llr_bad)

            self.rv_llr_interest = log_density(rv_llr_good) - log_density(rv_llr_bad)

            self.sample_llr_interest = sample(self.rv_llr_interest, size=100)

            self.lr_star = maximize(self.rv_llr_interest,
                    n_candidates = 100,
                    wrt_symbol = self.rv_llr,
                    wrt_init = self.sample_llr_interest)



class TestNested(unittest.TestCase):

    def setup(self):

        s_rng = tensor.shared_randomstreams.RandomStreams(23424)

        spec = self.spec = {}
        spec['lr'] = 10**s_rng.uniform((), low=-1, high=1)
        # make our prior on pp depend on lr
        spec['pp'] = s_rng.bernoulli((), p=1.0 / (1 + tensor.exp(-spec['lr'])))


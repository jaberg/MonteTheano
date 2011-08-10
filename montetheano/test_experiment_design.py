import tensor
from for_theano import where

class TestSimple(self):

    def setup(self):

        s_rng = tensor.shared_randomstreams.RandomStreams(23424)

        spec = self.spec = {}
        spec['lr'] = 10**s_rng.uniform((), low=0, high=1)
        spec['pp'] = s_rng.bernoulli((), p=0.5)

        self.data = [
                (.1, dict(lr=.7, pp=0)),
                (.2, dict(lr=.02, pp=1)),
                (.6, dict(lr=.0001, pp=1)),
                (.15, dict(lr=.001, pp=0)),
                (.01, dict(lr=.03, pp=0))]

    def test_algo1(self):

        # split data into good and bad
        self.data.sort()
        cutoff = int(.15 * len(self.data))
        good_data = self.data[:cutoff]
        bad_data = self.data[cutoff:]

        algo1 = Algo1()

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

    def test_rv_llr_interest(self):
        theano.printing.debugprint(self.rv_llr)#, 'asdf.png')


class TestNested(self):

    def setup(self):

        s_rng = tensor.shared_randomstreams.RandomStreams(23424)

        spec = self.spec = {}
        spec['lr'] = 10**s_rng.uniform((), low=-1, high=1)
        # make our prior on pp depend on lr
        spec['pp'] = s_rng.bernoulli((), p=1.0 / (1 + tensor.exp(-spec['lr'])))


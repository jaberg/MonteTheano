import numpy

import theano
from theano import tensor
from theano.printing import Print

from for_theano import where

class Algo1(self):

    def __init__(self, spec,
            good_fraction=.15,
            good_prior_weight=5,
            bad_prior_weight=5):
        self.spec = spec
        self.good_fraction = good_fraction
        self.good_prior_weight = good_prior_weight
        self.bad_prior_weight = bad_prior_weight

    def posterior_likelihood(self, spec, observations):
        """
        Rebuild the theano graph of the spec in the form of a posterior.
        """
        raise NotImplementedError()

    def suggest_trial(self, random_samples, losses):

        sorted_order = numpy.argsort(scores)

        del losses #losses = losses[sorted_order]

        # split data into good and bad
        good_samples = {}
        bad_samples = {}
        cutoff = int(self.good_fraction * len(sorted_order))
        for rv in random_samples:
            assert len(random_samples[rv]) == len(sorted_order)
            random_samples[rv] = random_samples[rv][sorted_order]

            good_samples[rv] = random_samples[rv][:cutoff]
            bad_samples[rv] = random_samples[rv][cutoff:]

        good_model = self.posterior_likelihood(good_samples)
        bad_model = self.posterior_likelihood(bad_samples)

        candidates = good_samples  # TODO: upsample w replacement
        interest = log(good_model) - log(bad_model)

        # maximize interest by piggy-backing on mcmc proposals and gradient info

        # return row of candidates with highest interest


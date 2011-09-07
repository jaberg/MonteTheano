
============
Monte Theano
============

Monte Carlo inference algorithms for stochastic Theano programs.

  1. Directed models: Use Theano (with RandomStreams) to build a directed graphical model, then

   - Estimate likelihood of a full assignment.

   - Condition on observations, draw samples from posterior over latent internal variables.

   - Estimate marginal likelihood analytically or by MCMC.

   - Learning by inferring MAP or ML estimates of latent variables.

  2. Undirected models: still thinking about if/how to do this. And what about
     factor graphs?



How does it work
----------------

Not totally clear yet!  Ingredients will be:

  - symbolic representations of likelihood functions

  - automatically factorizing directed models

  - generic Metropolis Hastings samplers

  - Hamiltonian Monte Carlo

  - Importance sampling?

  - Rejection sampling?

  - slice sampling?

  - Tempered sampling?

It seems like it should be possible to automatically recognize opportunities for
blocked Gibbs sampling, in which for example we recognize blocks of continuous
variables for an HMC sampler.  Not sure if this is a useful thing to do.


Similar Packages
----------------

  - MIT-Church (probabilistic scheme)

  - IBAL (probabilistic OCAML)

  - PyMC (MCMC inference in Python)

  - Infer.net (Csoft)

  - Factorie

  - PMTK

  - Dyna

This package differs from the ones above in building on top of Theano, which already has a) a
natural graph data structure for expressing directed graphical models, b) a
performance-oriented backend with GPU support, and c) automatic symbolic differentiation which
makes HMC and optimization routines much easier to implement.

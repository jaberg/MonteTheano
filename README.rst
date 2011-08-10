
============
Monte Theano
============

Monte Carlo inference algorithms for stochastic Theano programs.

  1. Use Theano (with RandomStreams) to build a directed graphical model.

  1. Use MonteTheano to do interesting things with that model:

   - Estimate likelihood of a full assignment.

   - Condition on observations, draw samples from latent internal variables.

   - Estimate likelihood of an incomplete assignment.


How does it work
----------------

This package implements various sampling strategies.


Similar Packages
----------------

  - MIT-Church (probabilistic scheme)

  - IBAL (probabilistic OCAML)

  - PyMC (MCMC inference in Python)

This package differs from the ones above in building on top of Theano, which already has a) a
natural graph data structure for expressing directed graphical models, b) a
performance-oriented backend with GPU support, and c) automatic symbolic differentiation which
makes HMC and optimization routines much easier to implement.


Dependencies
------------

Theano: requires `sorted_givens branch <https://github.com/jaberg/Theano/tree/sorted_givens>`_.

import numpy, pylab
import theano
from theano import tensor
from rstreams import RandomStreams
import distributions
from sample import mh2_sample
from for_theano import evaluate
from rv import full_log_likelihood

s_rng = RandomStreams(23424)

fair_prior = 0.999

coin_weight = tensor.switch(s_rng.binomial(1, fair_prior) > 0.5, 0.5, s_rng.dirichlet([1, 1])[0])

make_coin = lambda p, size: s_rng.binomial(1, p, draw_shape=(size,))    
coin = lambda size: make_coin(coin_weight, size)
            
for size in [1, 3, 6, 10, 20, 30, 50, 70, 100]:
    data = evaluate(make_coin(0.9, size))
            
    sampler = mh2_sample(s_rng, [coin_weight], {coin(size) : data})            
    
    print "nr of examples", size, ", estimated probability", sampler(nr_samples=400, burnin=20000, lag=10)[0].mean()

import numpy, pylab
import theano
from rstreams import RandomStreams
import distributions
from sample import mh2_sample
from for_theano import memoized

s_rng = RandomStreams(23424)

phi = s_rng.dirichlet(numpy.asarray([1, 1, 1, 1, 1]))
alpha = s_rng.gamma(2., 2.)        
prototype = phi*alpha

bag_prototype =  memoized(lambda bag: s_rng.dirichlet(prototype))
draw_marbles = lambda bag, nr: s_rng.multinomial(1, bag_prototype(bag), draw_shape=(nr,))

marbles_bag_1 = numpy.asarray([[1,1,1,1,1,1],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0]], dtype=theano.config.floatX).T                                
marbles_bag_2 = numpy.asarray([[0,0,0,0,0,0],
                               [1,1,1,1,1,1],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0]], dtype=theano.config.floatX).T 
marbles_bag_3 = numpy.asarray([[0,0,0,0,0,0],
                               [0,0,0,0,0,0],
                               [0,0,0,0,0,0],
                               [1,1,1,1,1,1],
                               [0,0,0,0,0,0]], dtype=theano.config.floatX).T 
marbles_bag_4 = numpy.asarray([[0],[0],[0],[0],[1]], dtype=theano.config.floatX).T 

givens = {draw_marbles(1,6): marbles_bag_1,
            draw_marbles(2,6): marbles_bag_2,
            draw_marbles(3,6): marbles_bag_3,
            draw_marbles(4,1): marbles_bag_4}
            
sampler = mh2_sample(s_rng, [draw_marbles(4,1)], givens)            

samples = sampler(200, 100, 100)
data = samples[0]

pylab.bar(range(5), data.sum(axis=0))
pylab.show()

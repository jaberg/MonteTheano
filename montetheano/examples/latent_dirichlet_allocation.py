import numpy
import theano
from theano import tensor
from rstreams import RandomStreams
import distributions
from sample import mh2_sample, mh_sample
from for_theano import memoized, evaluate

s_rng = RandomStreams(123)

nr_words = 4
nr_topics = 2
alpha = 0.8
beta = 1.

# Topic distribution per document
doc_mixture = memoized(lambda doc_id: s_rng.dirichlet([alpha/nr_topics]*nr_topics))

# Word distribution per topic
topic_mixture = memoized(lambda top_id: s_rng.dirichlet([beta/nr_words]*nr_words))

# For each word in the document, draw a topic according to multinomial with document specific prior
# TODO, see comment below: topics = memoized(lambda doc_id, nr: s_rng.multinomial(1, doc_mixture[doc_id], draw_shape=(nr,)))
topics = memoized(lambda doc_id, nr: s_rng.binomial(1, doc_mixture(doc_id)[0], draw_shape=(nr,)))

# Draw words for a specific topic
word_topic = lambda top_id: s_rng.multinomial(1, topic_mixture(top_id))

# TODO: memoized only works on the pre-compiled graph. This makes it fail in the case where we have to map 
# a vector of topics to individual multinomials with as priors the different topics. In the case of two topics
# we can hack around this by using a binomial topic distribution and using a switch statement here:
word_topic_mapper = lambda top_id: tensor.switch(top_id, word_topic(0), word_topic(1))

# Maps topics to words
# TODO, see comment above: get_words = memoized(lambda doc_id, nr: theano.map(word_topic, topics(doc_id, nr))[0])
get_words = memoized(lambda doc_id, nr: theano.map(word_topic_mapper, topics(doc_id, nr))[0])

# Define training 'documents'
document_1 = numpy.asarray([[1,0,0,0],
                            [1,0,0,0],
                            [0,1,0,0],
                            [1,0,0,0],
                            [0,1,0,0],
                            [0,1,0,0],
                            [1,0,0,0],
                            [0,1,0,0],
                            [1,0,0,0],
                            [0,1,0,0]], dtype=theano.config.floatX)

document_2 = numpy.asarray([[0,0,1,0],
                            [0,0,0,1],
                            [0,0,0,1],
                            [0,0,0,1],
                            [0,0,1,0],
                            [0,0,1,0],
                            [0,0,0,1],
                            [0,0,1,0],
                            [0,0,1,0],
                            [0,0,1,0]], dtype=theano.config.floatX)

document_3 = numpy.asarray([[1,0,0,0],
                            [0,0,0,1],
                            [0,1,0,0],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,1,0,0],
                            [0,0,0,1],
                            [1,0,0,0],
                            [0,0,1,0],
                            [0,0,1,0]], dtype=theano.config.floatX)

# Map documents to RVs
givens = {get_words(1, 10): document_1,
            get_words(2, 10): document_2,
            get_words(3, 10): document_3}

# Build sampler
sample, ll, updates = mh_sample(s_rng, [doc_mixture(1), doc_mixture(2), doc_mixture(3), topic_mixture(0), topic_mixture(1)])
sampler = theano.function([], sample, updates=updates, givens=givens, allow_input_downcast=True)

# Run sampling
for i in range(10000):
    d = sampler()            

    if i % 1000 == 0:
        print d

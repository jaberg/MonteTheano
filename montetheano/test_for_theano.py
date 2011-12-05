import numpy
import theano
from theano import tensor
from for_theano import infer_shape
import rstreams
import distributions
import for_theano

def test_infer_shape_const():
    shp = infer_shape(tensor.alloc(0, 5, 6, 7))
    print shp
    assert  shp == (5, 6, 7)

def test_infer_shape_shared_var():
    sv = theano.shared(numpy.asarray([2,3,5]))
    assert infer_shape(sv) == (3,)
    assert infer_shape(sv * 2 + 75) == (3,)

def test_shape_infer_shape():
    sv = theano.shared(numpy.asarray([2,3,5]))
    assert infer_shape(sv.shape) == (1,)

def test_shape_rv():
    R = tensor.shared_randomstreams.RandomStreams(234)
    n = R.normal(avg=0, std=1.0)
    assert infer_shape(n) == ()

def test_shape_scalar_rv_w_size():
    R = tensor.shared_randomstreams.RandomStreams(234)
    n = R.normal(avg=0, std=1.0, size=(40,20))
    assert infer_shape(n) == (40, 20)

def test_shape_scalar_rv_w_size_rstreams():
    R = rstreams.RandomStreams(234)
    n = R.normal(mu=0, sigma=1.0, draw_shape=(40,20))
    
    assert infer_shape(n) == (40, 20)

def test_shape_vector_rv_rstreams():
    R = rstreams.RandomStreams(234)
    n = R.normal(mu=numpy.zeros(10,), sigma=numpy.ones(10,), draw_shape=(10,))
    assert infer_shape(n) == (10,)

def test_shape_vector_rv_dirichlet_rstreams():
    R = rstreams.RandomStreams(234)
    n = R.dirichlet(alpha=numpy.ones(10,), draw_shape=(10,))
    assert infer_shape(n) == (10,), infer_shape(n)

def test_find():

    query = tensor.ivector()
    keepset = tensor.ivector()
    r = for_theano.find(query, keepset)

    assert r.ndim == 1
    assert 'int' in r.dtype

    f = theano.function([query, keepset], r)

    assert numpy.all(f([], []) == [])
    assert numpy.all(f([2, 1, 0, 4, 3], [5, 5, 5, 5]) == [])
    assert numpy.all(f([2, 1, 0, 4, 3], [4]) == [3])
    assert numpy.all(f([2, 1, 0, 4, 3], [4, 1]) == [1, 3])
    assert numpy.all(f([2, 1, 0, 4, 3], [1, 4]) == [1, 3])
    assert numpy.all(f([], [1, 4]) == [])

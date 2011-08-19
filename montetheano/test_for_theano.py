import numpy
import theano
from theano import tensor
from for_theano import infer_shape

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

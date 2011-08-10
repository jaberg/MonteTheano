import unittest
import theano
from gmm import AdaptiveParzen

class TestAdaptiveParzen(unittest.TestCase):
    def setUp(self):
        pass

    def test_0(self):
        # mu 0
        print theano.function([],
                AdaptiveParzen()([], 0, 1, 0.01))()

    def test_1(self):
        print theano.function([],
                AdaptiveParzen()([3.0], 0, 1, 0.01))()

    def test_2(self):
        print theano.function([],
                AdaptiveParzen()([3.0, 1.0], 0, 1, .01))()

    def test_3(self):
        print theano.function([],
                AdaptiveParzen()([0.01, 0.02, 0.003, 0.7], 0, 1, .01))()



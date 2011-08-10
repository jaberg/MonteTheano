import tensor

class TestSimple(self):

    def setup(self):

        s_rng = tensor.shared_randomstreams.RandomStreams(23424)

        spec = self.spec = {}
        spec['lr'] = 10**(s_rng.uniform(low=-5, high=-1))
        spec['pp'] = s_rng.bernoulli(p=.5)

        self.data = [
                (dict(lr=.002, pp=0), .1)
                (dict(lr=.02, pp=1), .2)
                (dict(lr=.0001, pp=1), .6)
                (dict(lr=.001, pp=0), .15)
                (dict(lr=.03, pp=0), .01)

    def 

import numpy
import theano


class Where(theano.Op):
    """
    Map a vector to an integer vector containing the sorted positions of
    non-zeros in the argument.

    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x):
        if x.ndim != 1:
            raise NotImplementedError()
        return theano.gof.Apply(self,
                [x],
                [tensor.lvector()])

    def perform(self, node, inputs, outstorage):
        outstorage[0][0] = numpy.where(inputs[0])
where = Where()


def multiswitch(*args):
    """Build a nested elemwise if elif ... statement.

        multiswitch(
            a, cond_a,
            b, cond_b,
            c)

    Translates roughly to an elementwise version of this...

        if cond_a:
            a
        elif cond_b:
            b
        else:
            c
    """
    assert len(args) % 2, 'need an add number of args'
    if len(args) == 1:
        return args[0]
    else:
        return tensor.switch(
                args[1],
                args[0],
                multiswitch(*args[2:]))

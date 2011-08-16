import copy
import numpy
import theano
from theano import tensor

def as_variable(thing):
    if isinstance(thing, theano.Variable):
        return thing
    if hasattr(thing, 'type'):
        return thing
    #TODO: why there is no theano.constant??
    return theano.shared(thing)


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


def elemwise_cond(*args):
    """Build a nested elemwise if elif ... statement.

        elemwise_cond(
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
                elemwise_cond(*args[2:]))


class LazySwitch(theano.gof.PureOp):
    """
    XXX

    """

    def __init__(self):
        pass

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self, other):
        return hash(type(self))

    def make_node(self, c, arg0, *args):
        for a in args:
            if a.type != arg0.type:
                raise TypeError(
                        'Switch requires same type for all cases',
                        (a.type, arg0.type))
        return theano.gof.Apply(self,
                [c, arg0] + list(args),
                [a.type()])

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        outtype = node.outputs[0].type
        c = node.inputs[0]
        s_output = node.outputs[0]
        ocontainer = storage_map[s_output]
        def thunk():
            if not compute_map[c][0]:
                return [0]  # ask to compute c
            else:
                casenum = storage_map[c][0]
                argvar = node.inputs[casenum+1]
                if compute_map[argvar][0]:
                    argval = storage_map[argvar][0]
                    ocontainer[0] = outtype.filter(
                            copy.deepcopy(argval))
                    return []  # computations are done
                else:
                    # ask to compute the input element we need
                    return [casenum+1]
        thunk.lazy = True
        thunk.inputs  = [storage_map[v] for v in node.inputs]
        thunk.outputs = [storage_map[v] for v in node.outputs]
        return thunk

lazy_switch = LazySwitch()

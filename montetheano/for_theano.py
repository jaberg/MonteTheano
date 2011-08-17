import numpy
import theano
from theano import tensor
from theano.gof import graph

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

def ancestors(variable_list, blockers = None):
    """Return the variables that contribute to those in variable_list (inclusive).

    :type variable_list: list of `Variable` instances
    :param variable_list:
        output `Variable` instances from which to search backward through owners
    :rtype: list of `Variable` instances
    :returns:
        all input nodes, in the order found by a left-recursive depth-first search
        started at the nodes in `variable_list`.

    """
    def expand(r):
        if r.owner and (not blockers or r not in blockers):
            l = list(r.owner.inputs)
            l.reverse()
            return l
    dfs_variables = graph.stack_search(graph.deque(variable_list), expand, 'dfs')
    return dfs_variables
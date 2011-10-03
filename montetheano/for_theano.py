import copy
import numpy
import theano
from theano import tensor
from theano.gof import graph

def evaluate(var):
    f = theano.function([], var, mode=theano.Mode(linker='py', optimizer=None))
    return f()

class memoized(object):
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        try:
            return self.cache[args]
        except KeyError:
            value = self.func(*args)                    
            self.cache[args] = value
            return value

def as_variable(thing, type=None):
    if isinstance(thing, theano.Variable):
        if type is None or thing.type == type:
            return thing
        else:
            raise TypeError(thing)
    if hasattr(thing, 'type'):
        if type is None or thing.type == type:
            return thing
        else:
            raise TypeError(thing)
    if type is None:
        #TODO: why there is no theano.constant??
        return theano.shared(thing)
    else:
        return type.Constant(
                type,
                type.filter(thing, allow_downcast=True))

class Bincount(theano.Op):
    """
    Map a vector to an integer vector containing the sorted positions of
    non-zeros in the argument.
    """
    #TODO: check to see if numpy.bincount supports minlength argument
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, weights=1, minlength=0):
        x = tensor.as_tensor_variable(x)
        weights = tensor.as_tensor_variable(weights)
        minlength = tensor.as_tensor_variable(minlength)
        if x.ndim != 1:
            raise NotImplementedError( x)
        if 'int' not in str(x.dtype):
            raise TypeError('bincount requires integer argument x', x)
        # TODO: check that weights and minlength are ok
        return theano.gof.Apply(self,
                [x, weights, minlength],
                [tensor.lvector()])

    def perform(self, node, inputs, outstorage):
        x, weights, minlength = inputs
        if weights == 1:
            rval = numpy.bincount(x)#, minlength=minlength)
        else:
            rval = numpy.bincount(*inputs)
        if len(rval) < minlength:
            tmp = numpy.zeros((minlength,), dtype=rval.dtype)
            tmp[:len(rval)] = rval
            rval = tmp
        outstorage[0][0] = rval

bincount = Bincount()


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
        outstorage[0][0] = numpy.asarray(numpy.where(inputs[0])[0])
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


class LazySwitch(theano.gof.op.PureOp):
    """
    lazy_switch(which_case, case0, case1, case2, case3, ...)

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


def clone_keep_replacements(i, o, replacements=None):
    """Duplicate nodes from i -> o inclusive.

    i - sequence of variables
    o - sequence of variables
    replacements - dictionary mapping each old node to its new one.
        (this is modified in-place as described in `clone_get_equiv`)

    By default new inputs are actually the same as old inputs, but
    when a replacements dictionary is provided this will not generally be the
    case.
    """
    equiv = clone_get_equiv(i, o, replacements)
    return [equiv[input] for input in i], [equiv[output] for output in o]


def clone_get_equiv(i, o, replacements=None):
    """Duplicate nodes from `i` to `o` inclusive.

    Returns replacements dictionary, mapping each old node to its new one.

    i - sequence of variables
    o - sequence of variables
    replacements - initial value for return value, modified in place.

    """
    if replacements is None:
	    d = {}
    else:
        d = replacements
    
    for input in i:
        if input not in d:
            d[input] = input
    
    for apply in graph.io_toposort(i, o):
        for input in apply.inputs:
            if input not in d:
                d[input] = input
        
        new_apply = apply.clone_with_new_inputs([d[i] for i in apply.inputs])
        if apply not in d:
            d[apply] = new_apply
        
        for output, new_output in zip(apply.outputs, new_apply.outputs):
            if output not in d:
                d[output] = new_output
    
    for output in o:
        if output not in d:
            d[output] = output.clone()
    
    return d
    
def evaluate_with_assignments(f, assignment):
    dfs_variables = ancestors([f], blockers=assignment.keys())
    frontier = [r for r in dfs_variables
            if r.owner is None or r in assignment.keys()]
    cloned_inputs, cloned_outputs = clone_keep_replacements(frontier, [f],
            replacements=assignment)
    out, = cloned_outputs
    
    return out
    
#
# SHAPE INFERENCE
#

# Shape.infer_shape
if not hasattr(theano.tensor.basic.Shape, 'infer_shape'):
    def shape_infer_shape(self, node, ishapes):
        return [(node.inputs[0].ndim,)]
    theano.tensor.basic.Shape.infer_shape = shape_infer_shape

# MakeVector.infer_shape
if not hasattr(theano.tensor.opt.MakeVector, 'infer_shape'):
    def makevector_infer_shape(self, node, ishapes):
        return [(node.inputs[0],)]
    theano.tensor.opt.MakeVector.infer_shape = makevector_infer_shape

def infer_shape_helper(v, assume_shared_size_fixed):
    if not isinstance(v.type, tensor.TensorType):
        return None

    if v.owner:
        if len(v.owner.outputs) > 1:
            output_pos = v.owner.outputs.index(v)
        else:
            output_pos = 0
        ishapes = [infer_shape_helper(i, assume_shared_size_fixed)
                for i in v.owner.inputs]
        return v.owner.op.infer_shape(v.owner, ishapes)[output_pos]


    if isinstance(v, theano.Constant):
        return v.data.shape

    if isinstance(v, theano.compile.SharedVariable):
        if assume_shared_size_fixed:
            return v.get_value(borrow=True).shape
        else:
            raise ValueError('shared var')

def infer_shape(v, assume_shared_size_fixed=True):
    rval = infer_shape_helper(v, assume_shared_size_fixed)
    if None is rval:
        raise TypeError('some ancestor was not a TensorType var')
    def as_int(o):
        if hasattr(o, 'data'):
            return int(o.data)
        elif hasattr(o, 'type'):
            f = theano.function([], o,
                    mode=theano.Mode(linker='py', optimizer=None))
            return f()
        else:
            return int(o)
    return tuple([as_int(r) for r in rval])



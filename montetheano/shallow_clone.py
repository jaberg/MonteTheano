import theano
from theano.gof import graph

def clone_keep_replacements(i, o, replacements=None):
    equiv = clone_get_equiv(i, o, replacements)
    return [equiv[input] for input in i], [equiv[output] for output in o]


def clone_get_equiv(i, o, replacements=None):
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

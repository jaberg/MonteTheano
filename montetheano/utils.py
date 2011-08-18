"""
Misc utils
"""

class ClobberContext(object):
    """
    Makes an object useable with 'with' statements.

    with obj as _:
        ... # obj.method is accessible as method()

    Danger - the illusion is not perfect! It works by inserting things into
    __builtin__ namespace, so if there are local variables in enclosing scopes,
    they will actually trump the object's own methods.
    """
    def __enter__(self):
        not hasattr(self, 'old')
        self.old = {}
        for name in self.clobber_symbols:
            if hasattr(__builtin__, name):
                self.old[name] = getattr(__builtin__, name)
            if hasattr(self.obj, name):
                setattr(__builtin__, name, getattr(self.obj, name))
        return self.obj

    def __exit__(self, e_type, e_val, e_traceback):
        for name in self.clobber_symbols:
            if name in self.old:
                setattr(__builtin__, name, self.old[name])
            elif hasattr(__builtin__, name):
                delattr(__builtin__, name)
        del self.old


class Updates(dict):
    """
    Updates is a dictionary for which the '+' operator does an update.

    Not a normal update though, because a KeyError is raised if a symbol is
    present in both dictionaries.
    """
    def __add__(self, other):
        rval = Updates(self)
        rval += other  # see: __iadd__
        return rval
    def __iadd__(self, other):
        d = dict(other)
        for k,v in d.items():
            if k in self and v != self[k]:
                raise KeyError()
            self[k] = v
        return self


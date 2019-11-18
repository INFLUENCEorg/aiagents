class Hashed:
    """
    Class that makes any object hashable by encapsulation.
    Assumes the object is not changed.
    If it is not immutable, and it is mutated,
    then the hashcode will become wrong.
    FIXME this should be in some utility directory 
    """

    def __init__(self, obj):
        self._obj = obj
        try:
            self._hash = hash(obj)
        except TypeError:
            self._hash = hash(str(obj))
            
    def get(self):
        """
        @return the original object
        """
        return self._obj
    
    def __eq__(self, other):
        return isinstance(other, Hashed) and self._obj == other._obj
    
    def __hash__(self):
        return self._hash

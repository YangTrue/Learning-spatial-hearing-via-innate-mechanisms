from collections import namedtuple

"""
    exercise book
    TODO read python lib doc, compare diff ways, namespace, dataclass and so on


    (pypbl) ➜  abide_vae-main git:(main) ✗ python
    Python 3.11.3 (main, May 15 2023, 15:45:52) [GCC 11.2.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> Dictionary1 = {'A': 'Geeks', 'B': 'For', 'C': 'Geeks'}
    >>> Dictionary1.keys()
    dict_keys(['A', 'B', 'C'])
    >>> Dictionary2 = {'D': 'Geeks', 'E': 'For', 'F': 'Geeks'}
    >>> Dictionary2.keys()
    dict_keys(['D', 'E', 'F'])
    >>> Dictionary1.keys()+Dictionary2.keys()
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    TypeError: unsupported operand type(s) for +: 'dict_keys' and 'dict_keys'
    >>> [x for x in Dictionary1.keys()]
    ['A', 'B', 'C']
    >>> # importing "collections" for namedtuple()
    >>> import collections
    >>>  
    >>> # Declaring namedtuple()
    >>> Student = collections.namedtuple('Student',
    ...                                  ['name', 'age', 'DOB'])
    >>>  
    >>> # Adding values
    >>> S = Student('Nandini', '19', '2541997')
    >>> s
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    NameError: name 's' is not defined. Did you mean: 'S'?
    >>> S
    Student(name='Nandini', age='19', DOB='2541997')
    >>> S._fields()
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    TypeError: 'tuple' object is not callable
    >>> S._fields
    ('name', 'age', 'DOB')
    >>> teacher=collections.namedtuple('teacher',['tid'])
    >>> t=teacher('123')
    >>> t._field
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    AttributeError: 'teacher' object has no attribute '_field'. Did you mean: '_fields'?
    >>> t._fields
    ('tid',)
    >>> S._fields+t._fields
    ('name', 'age', 'DOB', 'tid')
    >>> p=collections.namedtuple('person',S._fields+t._fields)
    >>> s1=Student._make(S._asdict())
    >>> s1
    Student(name='name', age='age', DOB='DOB')
    >>> Dictionary1+Dictionary2
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    TypeError: unsupported operand type(s) for +: 'dict' and 'dict'
    >>> Dictionary1.keys()&Dictionary2.keys()
    set()
    >>> isempty(Dictionary1.keys()&Dictionary2.keys())
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    NameError: name 'isempty' is not defined
    >>> o=Dictionary1.keys()&Dictionary2.keys()
    >>> o.isempty()
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    AttributeError: 'set' object has no attribute 'isempty'
    >>> Student
    <class '__main__.Student'>
    >>> Student._fiedls
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    AttributeError: type object 'Student' has no attribute '_fiedls'. Did you mean: '_fields'?
    >>> Student._fields
    ('name', 'age', 'DOB')
    >>> Student._fields & p._fields
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    TypeError: unsupported operand type(s) for &: 'tuple' and 'tuple'
    >>> 
"""

def is_noneoverlaping_dictionary(d1,d2):
    """
    return: 
        True if d1 and d2 has no overlapping keywords
    
    d.keys() return a set.
    & get the intersection
    """
    return len( d1.keys() & d2.keys())==0

def concatenate_namedtuples(n1, n2):
    """    
    assume none-overlapping keywords

    first loop through their keywords
    then copy the value
    """
    d1=n1._asdict()
    d2=n2._asdict()
    assert is_noneoverlaping_dictionary(d1,d2)
    
    # merge two dictionaries
    d3={**d1,**d2}

    # construct new namedtple
    newn=namedtuple('newn',n1._fields+n2._fields)

    # fill in the values
    #n3=newn._make(d3)
    n3=newn(**d3)

    return n3


if __name__=='__main__':
    s1 = namedtuple('Student1',['name', 'age', 'DOB'])('Nandini', '19', '2541997')
    s2 = namedtuple('Student2',['name2', 'age2', 'DOB2'])('Nandini2', '192', '25419972')
    s3= concatenate_namedtuples(s1,s2)
    print(s1,s2,s3)
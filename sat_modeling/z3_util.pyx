#cython: language_level=3, annotation_typing=True, embedsignature=True, boundscheck=True, wraparound=False, cdivision=True
#distutils: language = c++

cimport cython
from cython cimport view
from libc.stdlib cimport malloc, realloc, free
from libc.stdio cimport snprintf

from libcpp.vector cimport vector
from libcpp.string cimport string

from types cimport *

from cnf_util import CNF

from typing import List
from z3 import Bool

cdef extern from "<iostream>" namespace "std":
    cdef cppclass ostream:
        ostream& write(const char*, int) except +
        ostream& operator<<(bint val);
        ostream& operator<<(short val);
        ostream& operator<<(unsigned short val);
        ostream& operator<<(int val);
        ostream& operator<<(unsigned int val);
        ostream& operator<<(long val);
        ostream& operator<<(unsigned long val);
        ostream& operator<<(float val);
        ostream& operator<<(double val);
        ostream& operator<<(long double val);
        ostream& operator<<(void* val);

        ostream& operator<<(const char* val);
        ostream& operator<<(string val);


cdef extern from "<sstream>" namespace "std":
    cdef cppclass stringstream(ostream):
        ostream& write(const char*, int) except +
        string str() except +


def cnf_to_str(cnf: CNF, variables: List[Bool]):
    cdef size_t num_vars = len(variables)

    cdef uint64_t[:] _masks = cnf.masks
    cdef uint64_t[:] _signs = cnf.signs
    cdef size_t num_clauses = _signs.shape[0]
    cdef char *buf = NULL
    cdef size_t buflen = 0

    cdef vector[string] _variables
    cdef stringstream res
    cdef size_t i

    if _masks.shape[0] != _signs.shape[0]:
        raise ValueError('cnf.masks.shape != cnf.signs.shape')

    _variables.resize(num_vars)
    for i in range(num_vars):
        _variables[i] = variables[i].sexpr().encode()

    try:
        for i in range(num_vars):
            res << <char *> "(" << _variables[i] << <char *> "\n"
    finally:
        free(buf)

    return res.str()

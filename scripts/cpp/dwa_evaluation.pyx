"""
multiply.pyx

simple cython test of accessing a numpy array's data

the C function: c_multiply multiplies all the values in a 2-d array by a scalar, in place.

"""

import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern int c_dwa_evaluation (double* evalDB, double* trajDB, double* X, double* Vr, double* goal, double* obst, double* model, 
    double* evalParam, int oSz, double R, double dt)

@cython.boundscheck(False)
@cython.wraparound(False)
def dwa_evaluation(np.ndarray[double, ndim=2, mode="c"] evalDB not None, np.ndarray[double, ndim=2, mode="c"] trajDB not None,
    np.ndarray[double, ndim=1, mode="c"] X not None, np.ndarray[double, ndim=1, mode="c"] Vr not None,
    np.ndarray[double, ndim=1, mode="c"] goal not None, np.ndarray[double, ndim=2, mode="c"] obst not None,
    np.ndarray[double, ndim=1, mode="c"] model not None, np.ndarray[double, ndim=1, mode="c"] evalParam not None, 
    double R, double dt):

    cdef int oSz
    oSz = obst.shape[0]
    state = c_dwa_evaluation (&evalDB[0, 0], &trajDB[0, 0], &X[0], &Vr[0], &goal[0], &obst[0, 0], &model[0], &evalParam[0], oSz, R, dt)

    return state
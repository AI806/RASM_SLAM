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
cdef extern int c_bresenhamLine (double* mapMatrix, int* mapMatrixIndex, int* startPos, int* endPos, int xSize, int ySize, int pointCnt, double updateFree, double updateOcc, int currMarkFreeIndex, int currMarkOccIndex)

@cython.boundscheck(False)
@cython.wraparound(False)
def bresenhamLine(np.ndarray[double, ndim=2, mode="c"] mapMatrix not None, np.ndarray[int, ndim=2, mode="c"] mapMatrixIndex not None
    , np.ndarray[int, ndim=1, mode="c"] startPos not None,
    np.ndarray[int, ndim=2, mode="c"] endPos not None, double updateFree, double updateOcc, int currMarkFreeIndex, int currMarkOccIndex):
    """
    multiply (arr, value)

    Takes a numpy arry as input, and multiplies each elemetn by value, in place

    param: array -- a 2-d numpy array of np.float64
    param: value -- a number that will be multiplied by each element in the array

    """
    cdef int xSize, ySize, pointCnt, state

    xSize, ySize, pointCnt = mapMatrix.shape[0], mapMatrix.shape[1], endPos.shape[0]

    state = c_bresenhamLine (&mapMatrix[0,0], &mapMatrixIndex[0,0], &startPos[0], &endPos[0,0], xSize, ySize, pointCnt, updateFree, updateOcc, currMarkFreeIndex, currMarkOccIndex)

    return state
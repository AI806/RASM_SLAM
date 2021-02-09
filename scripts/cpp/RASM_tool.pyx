# distutils: language = c++

from RASM_tools cimport RASM_tools
cimport numpy as np

cdef class PyRASM_tools:
    cdef RASM_tools c_rasm_tool

    def __cinit__(self):
        self.c_rasm_tool = RASM_tools()

    def setLaserParam_p(self, l_sigma_hit, l_likelihood_max_dist, l_z_rand, r_max, l_z_hit, m_beam):
        self.c_rasm_tool.setLaserParam(l_sigma_hit, l_likelihood_max_dist, l_z_rand, r_max, l_z_hit, m_beam)

    def setMapParam_p(self, mSize_x, mapSz_y, mOffset_x, mOffset_y, r):
        self.c_rasm_tool.setMapParam(mSize_x, mapSz_y, mOffset_x, mOffset_y, r)

    def setpfParam_p(self, np.ndarray[double, ndim=1, mode="c"] rStd not None, np.ndarray[double, ndim=1, mode="c"] maStd not None, 
        np.ndarray[double, ndim=1, mode="c"] miStd not None, pCnt, mIter, cOptimalLoss, sLoss, o_stop_loss, cTimes, maRefCnt, miRefCnt):
        self.c_rasm_tool.setpfParam(&rStd[0], &maStd[0], &miStd[0], pCnt, mIter, cOptimalLoss, sLoss, o_stop_loss, cTimes, maRefCnt, miRefCnt)

    def getOptimalState_p(self, np.ndarray[double, ndim=1, mode="c"] state not None):
        cdef int status
        cdef double score
        self.c_rasm_tool.getOptimalState(&state[0], &status, &score)
        return status, score

    def refineRobotState_p(self,  np.ndarray[double, ndim=2, mode="c"] map not None, 
        np.ndarray[double, ndim=1, mode="c"] robotState not None, 
        np.ndarray[double, ndim=2, mode="c"] obstaclesToRobot not None):
        cdef int obsCnt = obstaclesToRobot.shape[0]
        cdef double ratio
        ratio = self.c_rasm_tool.refineRobotState(&map[0,0], &robotState[0], &obstaclesToRobot[0,0], obsCnt)
        return ratio

    def initProcess_p(self):
        self.c_rasm_tool.initProcess()

    def getOccMap_p(self, np.ndarray[double, ndim=2, mode="c"] map not None):
        self.c_rasm_tool.getOccMap(&map[0,0])

    def computeOccMap_p(self, np.ndarray[double, ndim=2, mode="c"] map not None):
        self.c_rasm_tool.computeOccMap(&map[0,0])

    def freeMem_p(self):
        self.c_rasm_tool.freeMem()

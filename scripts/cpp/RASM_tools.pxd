cdef extern from "RASM_tools.cpp":
    pass

# Declare the class with cdef
cdef extern from "RASM_tools.h":
    cdef cppclass RASM_tools:
        RASM_tools() except +

        void initProcess()

        void setLaserParam(double l_sigma_hit, double l_likelihood_max_dist, double l_z_rand, double r_max, double l_z_hit, int m_beam)
        void setMapParam(int mSize_x, int mSize_y, double mOffset_x, double mOffset_y, double r)
        void setpfParam(double *rStd, double *maStd, double *miStd, int pCnt, int mIter, double cOptimalLoss, double sLoss, 
            double o_stop_loss, int cTimes, int maRefCnt, int miRefCnt)

        void getOptimalState(double *state, int *status, double *score)
        void getOccMap(double *occMap)
        void computeOccMap(double *map)

        double refineRobotState(double *map, double *robotState, double *obstaclesToRobot, int obsCnt)
        void freeMem()
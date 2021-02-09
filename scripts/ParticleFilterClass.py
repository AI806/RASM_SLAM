import numpy as np
from itertools import *
import itertools
import time
import rospy
from scipy.stats import norm
import RASM_tool
import matplotlib.pyplot as plt

class ParticleFilter(object):

    def __init__(self, pfParams):

        # pStateCnt: the number of being estimated UWB nodes (m) and robots (n).  (m+n)*4, x, y, theta, v 
        # pfCnt: the number of particles,


        self.stateCnt = 3
        self.pfParams = pfParams
        np.random.seed(pfParams.seeds)
        self.mapSize = int(pfParams.mapSize / pfParams.resol)

        self.occMap = np.zeros((self.mapSize, self.mapSize))

        # self.particles = np.zeros((self.pfCnt, self.stateCnt))
        # self.weight = (np.ones((self.pfCnt, 1)) / self.pfCnt).flatten()
        # self.fStates = np.zeros((1, self.stateCnt))

        self.RASM_obj = RASM_tool.PyRASM_tools()
        self.RASM_obj.setLaserParam_p(pfParams.laser_sigma_hit, pfParams.laser_likelihood_max_dist, pfParams.laser_z_rand, \
            pfParams.range_max, pfParams.laser_z_hit, pfParams.maxBeam)
        self.RASM_obj.setMapParam_p(self.mapSize, self.mapSize, pfParams.mapOffset[0], pfParams.mapOffset[1], pfParams.resol)
        self.RASM_obj.setpfParam_p(np.array(pfParams.refineVar), np.array(pfParams.maxRefineStd), np.array(pfParams.minRefineStd),\
            pfParams.pRefineCnt, pfParams.maxIter, pfParams.optimal_loss, pfParams.stop_loss, pfParams.optimal_stop_loss, pfParams.checkTimes, \
            pfParams.pRefineMaxCnt, pfParams.pRefineMinCnt)
        self.RASM_obj.initProcess_p()

    def doParticleFiltering(self, mapMatrix5, posEst, obstaclesToRobot, vel, omega):

        start_time = time.time()
        self.RASM_obj.computeOccMap_p(mapMatrix5)
        optRobotSt = posEst.copy()
        obstaclesToRobot = obstaclesToRobot.copy(order='C')
        ## refining the robot state only based on the LiDAR observation, the state of UWB beacons keep unchanging at this step
        # the searching direction is based on the evaluate score at each iteration.
        ratio = self.RASM_obj.refineRobotState_p(mapMatrix5, optRobotSt, obstaclesToRobot)
        [pfState, score_ori] = self.RASM_obj.getOptimalState_p(optRobotSt)

        print("INFO: elapsed time ", time.time() - start_time, ", pfState, ", pfState, ", score_ori, ", score_ori, ", ratio, ", ratio)

        return pfState, optRobotSt, score_ori
    
    def freeMem(self):
        self.RASM_obj.freeMem_p()

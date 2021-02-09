import numpy as np
from itertools import *
import itertools
import time
import rospy
from scipy.stats import norm

class ParticleFilter(object):

    def __init__(self, pfParams):

        # pStateCnt: the number of being estimated UWB nodes (m) and robots (n).  (m+n)*4, x, y, theta, v 
        # pfCnt: the number of particles,
        self.pfCnt = pfParams.pfCnt
        self.resol = pfParams.resol #resolution of the map, default is 5 cm
        self.deltaT = pfParams.deltaT # the update time period
        self.mapOffset = pfParams.mapOffset
        self.pRefineCnt = pfParams.pRefineCnt

        self.refineVar = np.diag(pfParams.refineVar)
        self.maxIter = pfParams.maxIter
        self.threshold = pfParams.threshold
        self.mapSize = pfParams.mapSize
        self.alpha = pfParams.alpha
        self.overlapThres = pfParams.overlapThres
        self.lStop = pfParams.lStop

        self.stateCnt = 3
        np.random.seed(pfParams.seeds)

        self.xIdx = 0
        self.yIdx = 1
        self.thetaIdx = 2

        self.particles = np.zeros((self.pfCnt, self.stateCnt))
        self.weight = (np.ones((self.pfCnt, 1)) / self.pfCnt).flatten()
        self.fStates = np.zeros((1, self.stateCnt))

    def setParticleStates(self, pos):
        self.particles[:, :2] = np.tile(pos[:2].reshape(1, 2), (self.pfCnt, 1))
        self.particles[:, 2] = pos[-1]

    def getParticles(self):
        return self.particles.copy()

    def getEstPfState(self):
        return self.fStates.copy()

    def probToLog(self, prob):
        return np.log(prob / (1.0 - prob))

    def logToProb(self, prob):
        exp = np.exp(prob)
        return exp / (1 + exp)

    def sample(self, sigma):
        return (sigma + 1e-6) * np.random.randn(self.pfCnt, 1).flatten()
        
    def motionModel(self, particles, vel, omega): # in m

        absVel = np.abs(vel)
        absOmega = np.abs(omega)

        v_est = vel + self.sample(self.alpha[0]*absVel + self.alpha[1]*absOmega)
        o_est = omega + self.sample(self.alpha[2]*absVel + self.alpha[3]*absOmega)
        g_est = self.sample(self.alpha[4]*absVel + self.alpha[5]*absOmega)

        v_o = v_est / o_est

        deltTheta = o_est * self.deltaT
        ccos = np.cos(particles[:, self.thetaIdx])
        ssin = np.sin(particles[:, self.thetaIdx])
        dcos = np.cos(particles[:, self.thetaIdx] + deltTheta)
        dsin = np.sin(particles[:, self.thetaIdx] + deltTheta)
        
        particles[:, self.xIdx] = particles[:, self.xIdx] +\
             ( - v_o * ssin + v_o * dsin) 
        particles[:, self.yIdx] = particles[:, self.yIdx] +\
             ( v_o * ccos - v_o * dcos)
        particles[:, self.thetaIdx] = particles[:, self.thetaIdx] + \
            deltTheta + g_est * self.deltaT
        return particles
        
    # mapMatrix in 5cm, obstaclesToRobot in 5cm, particles in meter
    def evaluateLiDARWeight(self, mapMatrix5, obstaclesToRobot, particles, pfCnt):
        # obstaclesToRobot in 5cm, mapMatrix: one pixel is 5 cm

        # obstaclesInWorldCood: in 5cm, #           pfCnt   x   2   x   M 
        obstaclesInWorldCood = self.transformToWorldCoordinate(pfCnt, particles, obstaclesToRobot)
        # find the occupancy value for the individual projection of lidar ranges on each particles
        # the coordinate certer is at the center of the map, also convert the resolution of the coordinate from meter to 5 cm (map's resolution).
        mP5 = self.biLinearInterpolation(obstaclesInWorldCood, 3, mapMatrix5) # N x M

        objValue = (1 - mP5) ** 2
        objValue = np.mean(objValue, 1)# N x 1

        # # normalize the loss, the bigger the less aligned.
        # weight = objValue / np.sum(objValue.flatten())
        return objValue

    def biLinearInterpolation(self, obstacles, flag, mapMatrix5): # flag 1:3D, 2:2D, obstacles: 2 x M
        
        mapSize = self.mapSize / self.resol # in 5cm
        if flag == 2:
            
            lbPoints = obstacles.astype(int)
            rpPoints = lbPoints + 1
            lbPoints[lbPoints >= mapSize] = mapSize - 1
            lbPoints[lbPoints < 0] = 0
            rpPoints[rpPoints >= mapSize] = mapSize - 1
            rpPoints[rpPoints < 0] = 0
            # P00, P01, P10, P11
            mapValue = self.logToProb(np.array([mapMatrix5[lbPoints[1, :], lbPoints[0, :]], \
                mapMatrix5[rpPoints[1, :], lbPoints[0, :]], \
                mapMatrix5[lbPoints[1, :], rpPoints[0, :]], \
                mapMatrix5[rpPoints[1, :], rpPoints[0, :]]]).T)

            mapValue[np.isnan(mapValue)] = 1

            p_p0 = obstacles - lbPoints # [x - x0, y - y0]
            p1_p = rpPoints - obstacles # [x1 - x, y1 - y]

            mP = p_p0[1, :] * (p_p0[0, :] * mapValue[:, 3] + p1_p[0, :] * mapValue[:, 1]) + \
                p1_p[1, :] * (p_p0[0, :] * mapValue[:, 2] + p1_p[0, :] * mapValue[:, 0])
            return mP
        elif flag == 1:
            lbPoints = obstacles.astype(int)
            rpPoints = lbPoints + 1
            p_p0 = obstacles - lbPoints # [x - x0, y - y0]
            p1_p = rpPoints - obstacles # [x1 - x, y1 - y]
            lbPoints[lbPoints >= mapSize] = mapSize - 1
            lbPoints[lbPoints < 0] = 0
            rpPoints[rpPoints >= mapSize] = mapSize - 1
            rpPoints[rpPoints < 0] = 0

            P00 = mapMatrix5[lbPoints[:, 1, :], lbPoints[:, 0, :]]
            P01 = mapMatrix5[rpPoints[:, 1, :], lbPoints[:, 0, :]]
            P10 = mapMatrix5[lbPoints[:, 1, :], rpPoints[:, 0, :]]
            P11 = mapMatrix5[rpPoints[:, 1, :], rpPoints[:, 0, :]]
            
            P00 = self.logToProb(P00)
            P01 = self.logToProb(P01)
            P10 = self.logToProb(P10)
            P11 = self.logToProb(P11)

            mP = p_p0[:, 1, :] * (p_p0[:, 0, :] * P11 + p1_p[:, 0, :] * P01) + \
                p1_p[:, 1, :] * (p_p0[:, 0, :] * P10 + p1_p[:, 0, :] * P00)
            return mP
        elif flag == 3: # no interpolitation
            # resolution 5cm, obstacles: N x 2 x M
            obstaclesTmp = (obstacles + 0.5).astype(int)
            obstaclesTmp[obstaclesTmp >= mapSize] = mapSize - 1
            obstaclesTmp[obstaclesTmp <= 0] = 0
            mP5 = mapMatrix5[obstaclesTmp[:, 1, :], obstaclesTmp[:, 0, :]]
            mP5 = self.logToProb(mP5)
            mP5[np.isnan(mP5)] = 1

            return mP5

    def transformToWorldCoordinate(self, pCnt, robotState, obstaclesToRobot):

        theta = robotState[:, -1]
        # compute the rotate matrix
        ccos = np.cos(theta)
        ssin = np.sin(theta)
        Ro = np.zeros((pCnt, 2, 2))
        Ro[:, 0, 0] = ccos
        Ro[:, 0, 1] = -ssin
        Ro[:, 1, 0] = ssin
        Ro[:, 1, 1] = ccos
        robotPos = robotState[:, :2].reshape(pCnt, 2, 1) # N   x   2   x   1   
        obstacles = np.dot(Ro, obstaclesToRobot) + np.tile(robotPos, (1, 1, obstaclesToRobot.shape[1]))
        #           N   x   2   x   M               N   x   2   x   M 
        return obstacles

    # robotState in 5cm and need to be offsetted to the map center, mapMatrix in 5cm, obstaclesToRobot in 5 cm
    def computeTheLossOnLidar(self, mapMatrix, robotState, obstaclesToRobot):

        robotState = robotState.reshape(1, self.stateCnt)
        obstaclesInWorldCood = self.transformToWorldCoordinate(1, robotState, obstaclesToRobot) # 1 x 2 x M
        obstaclesInWorldCood = obstaclesInWorldCood.reshape(obstaclesInWorldCood.shape[1], obstaclesInWorldCood.shape[2])

        # bilinear interpolation
        mapOccProb = self.biLinearInterpolation(obstaclesInWorldCood, 2, mapMatrix)

        # compute the loss for lidar scanning
        objValue = (1 - mapOccProb) ** 2
        loss = np.mean(objValue) # 1 x 1

        freeCnt = np.sum(mapOccProb < 0.5)
        occCnt = np.sum(mapOccProb > 0.5)

        score = freeCnt / float(occCnt + 1.)
        scoreValid = (occCnt) / float(obstaclesInWorldCood.shape[1])
        return score, scoreValid, loss

    # posState: position in meter, var in meter, pCnt: count of particles
    def particleStateUpdate(self, posState, var, pCnt): ## var[:, 0] x, var[:, 1] y, var[:, 2] heading  # x, y, theta, vel

        particles = np.random.multivariate_normal(posState, var, pCnt)
        particles[:, 0] = particles[:, 0] / self.resol + self.mapOffset[0] # convert to 5cm
        particles[:, 1] = particles[:, 1] / self.resol + self.mapOffset[1] # convert to 5cm

        return particles
    
    # this function is to find the weight for each particle according to the position estimated from EKF (using UWB range table)
    def evaluateWeightForPos(self, posEst):
        dist = (posEst[0] - self.particles[:, 0]) ** 2 + (posEst[1] - self.particles[:, 1]) ** 2 
        sigma = 0.3 ** 2
        wD = np.exp( - dist / (sigma * 2) )
        wD = wD / np.sum(wD)

        vectorA = np.tile(np.array([ np.cos(posEst[2]), np.sin(posEst[2])]).reshape(1, 2), (self.pfCnt, 1) )
        vectorB = np.array([ np.cos(self.particles[:, 2]), np.sin(self.particles[:, 2])]).T

        num = np.sum( vectorA * vectorB, axis=1 )
        den = np.linalg.norm(vectorA, axis=1) * np.linalg.norm(vectorB, axis = 1)
        theta = np.arccos(num / den) # 0 - pi
        wH = theta * theta

        sigma = (5 * np.pi / 180.0) ** 2
        wH = np.exp( - wH / (sigma * 2) )
        wH = wH / np.sum(wH)

        return wD * wH

    def updateVariance(self, lT1, lT2, preSigma):
        
        rho = (lT2 - lT1) / (lT1 - self.lStop + 1e-5)

        if rho > 5:
            rho = 5
        if rho < 0.2:
            rho = 0.2
        print("INFO: lT1, ", lT1, ", lT2, ", lT2, ", rho, ", rho)

        sigma = rho * preSigma
        if sigma[0][0] > 0.5:
            sigma[0][0] = 0.5
        elif sigma[0][0] < 0.01:
            sigma[0][0] = 0.01

        if sigma[1][1] > 0.5:
            sigma[1][1] = 0.5
        elif sigma[1][1] < 0.01:
            sigma[1][1] = 0.01

        if sigma[2][2] > 0.5:
            sigma[2][2] = 0.5
        elif sigma[2][2] < 0.01:
            sigma[2][2] = 0.01

        return sigma

    # self.particles in meter, mapMatrix in 5cm. posEst in meter, obstaclesToRobot in 5 cm, vel_r and vel_l in 1cm, 
    def doParticleFiltering_1(self, mapMatrix5, mapMatrixSave, \
        posEst, obstaclesToRobot, vel, omega):
        
        ## particle filter for coarse state estimation
        # update the motion state for each of the particles
        self.particles = self.motionModel(self.particles, vel, omega)
        # coarsely estimate the state of the robot's state according to the estmated position of robot from EKF
        self.weight = self.weight * self.evaluateWeightForPos(posEst)
        sumW = np.sum(self.weight)
        if sumW < 1e-10:
            self.weight = 1 / float(self.pfCnt)
        else:
            self.weight = self.weight / (sumW)
        # compute the coarse pose of the robot, in meter
        
        coarsePos = np.sum(self.particles * np.tile(self.weight.reshape(self.pfCnt, 1), (1, 3)), 0)
        coarsePosInMap = coarsePos.copy()
        coarsePosInMap[:2] = coarsePosInMap[:2] / self.resol + self.mapOffset # convert to 5cm and move to map coordinate

        ## refining the robot state only based on the LiDAR observation, the state of UWB beacons keep unchanging at this step
        # the motion state of each particles is update according to their changes on the loss function
        [score, validScore, curLoss] = self.computeTheLossOnLidar(mapMatrix5, coarsePosInMap, obstaclesToRobot)

        # initial parameter for pose refinement
        varianceNew = self.refineVar.copy()
        bestLoss = curLoss
        preLoss = curLoss
        fRobotStates = coarsePos.copy() # in meter
        finalStates = coarsePos.copy()
        iterCnt = 0
        
        while iterCnt < self.maxIter and not rospy.is_shutdown(): 
            # the loss not meets the requirement of user defined
            particle = self.particleStateUpdate(fRobotStates, varianceNew, self.pRefineCnt) # in meter

            # Note that the particles here will be moved to map center and conver to 5cm even if without the return of particles
            weights = self.evaluateLiDARWeight(mapMatrix5, obstaclesToRobot, particle, self.pRefineCnt) 
        
            # find the minimum weight
            idx = np.where(weights == np.min(weights))
            fRobotStates = particle[idx[0][0], :].copy() # note there need the copy of fRobotStates since fRobotStates will keep changing in the later iteration 
            [score, validScore, curLoss] = self.computeTheLossOnLidar(mapMatrix5, fRobotStates, obstaclesToRobot)

            fRobotStates[:2] = (fRobotStates[:2] - self.mapOffset) * self.resol  # move to the coordinate of UWB map and convert to meter
            if bestLoss > curLoss:
                bestLoss = curLoss
                # varianceNew = self.updateVariance(bestLoss, preLoss, varianceNew)
                varianceNew = varianceNew * 0.5
                preLoss = bestLoss
                finalStates = fRobotStates.copy() # note there need the copy of fRobotStates since fRobotStates will keep changing in the later iteration 
                if curLoss < 0.15:
                    break
            iterCnt += 1

        pfState = 1 #Not find the optimal one, donot update the particles and particles are not updated

        print("INFO: bestLossrobotState, ", bestLoss)
        if (bestLoss < 0.15): #successfully find the optimal point
            pfState = 2
            # update the particles
            self.setParticleStates(finalStates.flatten())
            self.weight = (np.ones((self.pfCnt, 1)) / self.pfCnt).flatten()
        else:
            finalStates = coarsePos.copy()

        self.fStates = finalStates.copy()
        # check if resampling needed
        if 1. / np.sum(self.weight ** 2) < self.pfCnt / 2.:  # If particle cloud degenerate:
            indx = np.random.choice(self.pfCnt, self.pfCnt, p = self.weight)
            self.particles = self.particles[indx, :]# .reshape(self.pfCnt, 4)
            self.weight = (np.ones((self.pfCnt, 1)) / self.pfCnt).flatten()

        return pfState, finalStates

    def doParticleFiltering(self, mapMatrix5, mapMatrixSave, \
        posEst, obstaclesToRobot, vel, omega):
        start_time = time.time()
        ## particle filter for coarse state estimation
        # update the motion state for each of the particles
        self.particles = self.motionModel(self.particles, vel, omega)
        # coarsely estimate the state of the robot's state according to the estmated position of robot from EKF
        self.weight = self.weight * self.evaluateWeightForPos(posEst)
        sumW = np.sum(self.weight)
        if sumW < 1e-10:
            self.weight = 1 / float(self.pfCnt)
        else:
            self.weight = self.weight / (sumW)
        # compute the coarse pose of the robot, in meter
        coarsePos = np.sum(self.particles * np.tile(self.weight.reshape(self.pfCnt, 1), (1, 3)), 0)
        coarsePosInMap = coarsePos.copy()
        coarsePosInMap[:2] = coarsePosInMap[:2] / self.resol + self.mapOffset # convert to 5cm and move to map coordinate

        ## refining the robot state only based on the LiDAR observation, the state of UWB beacons keep unchanging at this step
        # the motion state of each particles is update according to their changes on the loss function
        [score, validScore, curLoss] = self.computeTheLossOnLidar(mapMatrix5, coarsePosInMap, obstaclesToRobot)

        # initial parameter for pose refinement
        varianceNew = self.refineVar.copy()
        bestLoss = curLoss
        fRobotStates = coarsePos.copy() # in meter
        finalStates = coarsePos.copy()
        preBestState = coarsePos.copy()
        iterCnt = 0
        bestScore = score
        bestValidScore = validScore

        
        while iterCnt < self.maxIter and not rospy.is_shutdown(): 
            # the loss not meets the requirement of user defined
            particle = self.particleStateUpdate(fRobotStates, varianceNew, self.pRefineCnt) # in meter

            # Note that the particles here will be moved to map center and conver to 5cm even if without the return of particles
            weights = self.evaluateLiDARWeight(mapMatrix5, obstaclesToRobot, particle, self.pRefineCnt) 
        
            # find the minimum weight
            idx = np.where(weights == np.min(weights))
            fRobotStates = particle[idx[0][0], :].copy() # note there need the copy of fRobotStates since fRobotStates will keep changing in the later iteration 
            [score, validScore, curLoss] = self.computeTheLossOnLidar(mapMatrix5, fRobotStates, obstaclesToRobot)

            fRobotStates[:2] = (fRobotStates[:2] - self.mapOffset) * self.resol  # move to the coordinate of UWB map and convert to meter
            if bestLoss > curLoss:
                varianceNew[:2] = varianceNew[:2] * 0.8
                varianceNew[2] = varianceNew[2] * 0.8
                bestLoss = curLoss
                preBestState = fRobotStates.copy()
                finalStates = fRobotStates.copy() # note there need the copy of fRobotStates since fRobotStates will keep changing in the later iteration 
                bestScore = score
                bestValidScore = validScore
            iterCnt += 1

        
        pfState = 1 #Not find the optimal one, donot update the particles and particles are not updated
        if (bestScore < self.threshold and bestValidScore > self.overlapThres): #successfully find the optimal point
            pfState = 2
            # update the particles
            # Weight = self.evaluateWeightForPos(fStates)
            # self.weight = self.weight * Weight
            # self.weight = self.weight / np.sum(self.weight)
            self.setParticleStates(finalStates.flatten())
            self.weight = (np.ones((self.pfCnt, 1)) / self.pfCnt).flatten()
        else:
            finalStates = coarsePos.copy()

        self.fStates = finalStates.copy()
        # check if resampling needed
        if 1. / np.sum(self.weight ** 2) < self.pfCnt / 2.:  # If particle cloud degenerate:
            indx = np.random.choice(self.pfCnt, self.pfCnt, p = self.weight)
            self.particles = self.particles[indx, :]# .reshape(self.pfCnt, 4)
            self.weight = (np.ones((self.pfCnt, 1)) / self.pfCnt).flatten()

        print("INFO: elapsed time, ", time.time() - start_time)
        return pfState, finalStates
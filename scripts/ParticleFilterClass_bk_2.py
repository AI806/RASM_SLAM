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
        self.fWheelDistance = pfParams.fWheelDistance # the distance between two wheels
        self.deltaT = pfParams.deltaT # the update time period
        self.mapOffset = pfParams.mapOffset
        self.pRefineCnt = pfParams.pRefineCnt

        self.refineVar = np.diag(pfParams.refineVar)
        self.maxIter = pfParams.maxIter
        self.threshold = pfParams.threshold
        self.visualizeParticle = pfParams.visualizeParticle
        self.pValue = pfParams.pValue
        self.step = pfParams.iterStepPvalue
        self.maxIterPvalue = pfParams.maxIterPvalue
        self.mapSize = pfParams.mapSize
        self.alpha = pfParams.alpha
        self.overlapThres = pfParams.overlapThres

        self.stateCnt = 3
        np.random.seed(pfParams.seeds)

        self.xIdx = 0
        self.yIdx = 1
        self.thetaIdx = 2

        self.particles = np.zeros((self.pfCnt, self.stateCnt))
        self.weightD = (np.ones((self.pfCnt, 1)) / self.pfCnt).flatten()
        self.weightH = (np.ones((self.pfCnt, 1)) / self.pfCnt).flatten()
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

    def motionModel_1(self, particles, var, vel_l, vel_r): # in m

        # Motion update for the states of UWB nodes on the robot.
        newWheelVel_l = var[0]*np.random.randn(self.pfCnt, 1) + (vel_l)
        newWheelVel_r = var[1]*np.random.randn(self.pfCnt, 1) + (vel_r)
        # newWheelVel_l = np.tile(vel_l, (self.pfCnt, 1)) / 100.0
        # newWheelVel_r = np.tile(vel_r, (self.pfCnt, 1)) / 100.0

        angularVel = ((newWheelVel_r - newWheelVel_l) / self.fWheelDistance)
        deltaAngular = angularVel * self.deltaT
        iccRadius = 0.5 * (newWheelVel_l + newWheelVel_r).flatten() / (angularVel.flatten())

        ccos = np.cos(particles[:, self.thetaIdx]).reshape(1, self.pfCnt)
        ssin = np.sin(particles[:, self.thetaIdx]).reshape(1, self.pfCnt)

        iccPosx = particles[:, self.xIdx] - iccRadius * ssin
        iccPosy = particles[:, self.yIdx] + iccRadius * ccos

        ccos = np.cos(deltaAngular).reshape(1, self.pfCnt)
        ssin = np.sin(deltaAngular).reshape(1, self.pfCnt)
        deltaX = particles[:, self.xIdx] - iccPosx
        deltaY = particles[:, self.yIdx] - iccPosy

        particles[:, self.xIdx] = ccos*deltaX - ssin*deltaY + iccPosx
        particles[:, self.yIdx] = ssin*deltaX + ccos*deltaY + iccPosy

        particles[:, self.thetaIdx] = particles[:, self.thetaIdx] + deltaAngular.T
        return particles

    def sample(self, sigma):
        return (sigma + 1e-6) * np.random.randn(self.pfCnt, 1).flatten()
        

    def motionModel_bk(self, particles, vel, omega): # in m

        # vel = (vel_l + vel_r) / 2 
        # omega = (vel_r - vel_l) / (self.fWheelDistance * 100.0)

        # absVel = np.abs(vel)
        # absOmega = np.abs(omega)

        absVel = vel
        absOmega = omega

        v_est = vel + self.sample(self.alpha[0]*absVel + self.alpha[1]*absOmega)
        o_est = omega + self.sample(self.alpha[2]*absVel + self.alpha[3]*absOmega)
        g_est = self.sample(self.alpha[4]*absVel + self.alpha[5]*absOmega)

        deltTheta = o_est * self.deltaT
        ccos = np.cos(particles[:, self.thetaIdx])
        ssin = np.sin(particles[:, self.thetaIdx])
        dcos = np.cos(particles[:, self.thetaIdx] + deltTheta)
        dsin = np.sin(particles[:, self.thetaIdx] + deltTheta)

        particles[:, self.xIdx] = particles[:, self.xIdx] +\
             ( - v_est * ssin + v_est * dsin) / o_est 
        particles[:, self.yIdx] = particles[:, self.yIdx] +\
             ( v_est * ccos - v_est * dcos) / o_est
        particles[:, self.thetaIdx] = particles[:, self.thetaIdx] + \
            deltTheta + g_est * self.deltaT
        return particles

    def motionModel(self, particles, vel, omega): # in m

        absVel = vel * vel 
        absOmega = omega * omega

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
        # obstaclesToRobot in 5cm, mapMatrix: one pixel is 5 cm, particles: in 5cm, need to convert to 5 cm

        particles[:, :2] = particles[:, :2] / self.resol + self.mapOffset # convert to 5cm
        # obstaclesInWorldCood: in 5cm
        obstaclesInWorldCood = self.transformToWorldCoordinate(pfCnt, particles, obstaclesToRobot)
        # find the occupancy value for the individual projection of lidar ranges on each particles
        # the coordinate certer is at the center of the map, also convert the resolution of the coordinate from meter to 5 cm (map's resolution).
        start_time = time.time()
        # mapOccProb = self.biLinearInterpolation(obstaclesInWorldCood, 3, mapMatrix5) # N x M
        [mP20, mP10, mP5] = self.biLinearInterpolation(obstaclesInWorldCood, 3, mapMatrix5) # N x M
        
        objValue20 = 1
        objValue10 = 1

        objValue5 = (1 - mP5) ** 2
        objValue5 = np.mean(objValue5, 1) + 1e-5 # N x 1

        objValue = objValue20 * objValue10 * objValue5
        # # normalize the loss, the bigger the less matched.
        # weight = objValue / np.sum(objValue.flatten())
        return objValue, particles

    def biLinearInterpolation(self, obstacles, flag, mapMatrix5): # flag 1:3D, 2:2D, obstacles: 2 x M
        
        mapSize = self.mapSize / 0.05 # in 5cm
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

            mapSize = mapSize * 5 # in 1cm
            # obstacles in 1cm
            obstacles = obstacles * 5
            # limit the obstacles' space
            obstacles[obstacles >= mapSize] = mapSize - 1
            obstacles[obstacles < 0] = 0

            mP20 = 1
            mP10 = 1

            # resolution 50cm
            obstaclesTmp = (obstacles / 5.0).astype(int)
            mP5 = mapMatrix5[obstaclesTmp[:, 1, :], obstaclesTmp[:, 0, :]]
            mP5 = self.logToProb(mP5)
            mP5[np.isnan(mP5)] = 1

            return mP20, mP10, mP5

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
        robotPos = robotState[:, :2].reshape(pCnt, 2, 1) #N   x   2   x   1   
        obstacles = np.dot(Ro, obstaclesToRobot) + np.tile(robotPos, (1, 1, obstaclesToRobot.shape[1]))
        #           N   x   2   x   M               N   x   2   x   M 
        return obstacles

    # robotState in 5cm and need to be offsetted to the map center, mapMatrix in 5cm, obstaclesToRobot in 5 cm
    def computeTheLossOnLidar(self, mapMatrix, robotState, obstaclesToRobot):

        robotState = robotState.reshape(1, self.stateCnt) 
        obstaclesInWorldCood = self.transformToWorldCoordinate(1, robotState, obstaclesToRobot)
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

        # print scoreValid
        return score, scoreValid, loss

    # posState: position in meter, var in meter, pCnt: count of particles
    def particleStateUpdate(self, posState, var, pCnt): ## var[:, 0] x, var[:, 1] y, var[:, 2] heading  # x, y, theta, vel

        particles = np.random.multivariate_normal(posState, var, pCnt)
        return particles
    
    # this function is to find the weight for each particle according to the position estimated from EKF (using UWB range table)
    def evaluateWeightForPos(self, posEst):
        dist = np.sqrt( (posEst[0] - self.particles[:, 0]) ** 2 + (posEst[1] - self.particles[:, 1]) ** 2 )

        vectorA = np.tile(np.array([ np.cos(posEst[2]), np.sin(posEst[2])]).reshape(1, 2), (self.pfCnt, 1) )
        vectorB = np.array([ np.cos(self.particles[:, 2]), np.sin(self.particles[:, 2])]).reshape(self.pfCnt, 2)
        # wH = 1 - np.sum(vectorA * vectorB, 1)

        num = np.sum( vectorA * vectorB, axis=1 )
        den = np.linalg.norm(vectorA, axis=1) * np.linalg.norm(vectorB, axis = 1)
        wH = num / den
        # wH = (np.cos(posEst[2]) * np.cos(self.particles[:, 2]) + np.sin(posEst[2]) * np.sin(self.particles[:, 2])) + 1

        wD = np.exp( - dist * 2)
        wD = wD / np.sum(wD)
        wH = wH / np.sum(wH)
        
        # w = wD * wH
        # w = w / np.sum(w)
        # w = 1
        return wD, wH
    
    def estimateNewVariance(self, preBestState, fRobotStates, preSigma):
        
        destDist = np.abs( preBestState - fRobotStates )
        sigma = preSigma.copy()
        idx = 0
        for i in np.arange(len(preSigma)):
            dest = -norm.ppf(self.pValue, 0, sigma[i, i])
            while dest - destDist[i] > 0.01:
                s = 0.2 * np.abs(dest - destDist[i])
                if s < self.step:
                    s = self.step
                sigma[i, i] -= s
                dest = -norm.ppf(self.pValue, 0, sigma[i, i])

                # in case that the program cannot go out the loop
                idx += 1
                if idx > self.maxIterPvalue:
                    sigma = preSigma
                    break
        return sigma

    # self.particles in meter, mapMatrix in 5cm. posEst in meter, obstaclesToRobot in 5 cm, vel_r and vel_l in 1cm, 
    def doParticleFiltering(self, mapMatrix5, mapMatrixSave, \
        posEst, obstaclesToRobot, vel, omega):
        
        # update the motion state for each of the particles
        self.particles = self.motionModel(self.particles, vel, omega) # 1 for particle update with odometry, 2 for particle update without odo
        # coarsely estimate the state of the robot's state according to the estmated position of robot from EKF
        [uwbWeight, headingWeight] = self.evaluateWeightForPos(posEst)
        self.weightD = self.weightD * uwbWeight
        self.weightD = self.weightD / np.sum(self.weightD)
        self.weightH = self.weightH * headingWeight
        self.weightH = self.weightH / np.sum(self.weightH)
        
        # print self.weight
        coarsePos = np.zeros((3, 1)).flatten()
        uwbPosEst = self.particles[:, :2] * np.tile(self.weightD.reshape(self.pfCnt, 1), (1, 2))
        headingEst = self.particles[:, 2] * self.weightH
        coarsePos[:2] = np.sum(uwbPosEst, 0) # in meter
        coarsePos[2] = np.sum(headingEst)
        # if coarsePos[2] < 0:
        #     coarsePos[2] = coarsePos[2] + 2 * np.pi
        # elif coarsePos[2] > 2*np.pi:
        #     coarsePos[2] = coarsePos[2] - 2*np.pi
        
        # coarsePos[:2] = posEst[:2]
        # coarsePos[2] = posEst[-1]

        coarsePosInMap = coarsePos.copy()
        coarsePosInMap[:2] = coarsePosInMap[:2] / self.resol + self.mapOffset # convert to 5cm and move to map center
        # then refine the robot state only based on the LiDAR observation, the state of UWB beacons keep unchanging at this step
        # the motion state of each particles is update according to their changes on the loss function
        # curLoss = self.computeTheLossOnLidar(mapMatrix5, coarsePosInMap, obstaclesToRobot)
        [score, validScore, curLoss] = self.computeTheLossOnLidar(mapMatrix5, coarsePosInMap, obstaclesToRobot)

        varianceNew = self.refineVar.copy()
        bestLoss = curLoss
        pfStateiterCnt = 0
        start_time = time.time()
        updateUwb = False
        fStates = coarsePos.copy() # in meter
        finalStates = coarsePos.copy()
        preBestState = coarsePos.copy()
        iterCnt = 0
        bestScore = score
        bestValidScore = validScore
        while iterCnt < self.maxIter and not rospy.is_shutdown(): 
            # the loss not meets the requirement of user defined
            # particles = self.motionModel(particles, varianceNew, 0, 0)
            particle = self.particleStateUpdate(fStates, varianceNew, self.pRefineCnt) # in meter
            # Note that the particles here will be moved to map center and conver to 5cm even if without the return of particles
            [weights, particle] = self.evaluateLiDARWeight(mapMatrix5, obstaclesToRobot, particle, self.pRefineCnt) 
        
            # find the maxinum weigt
            minW = np.min(weights)
            idx = np.where(weights == minW)
            if len(idx[0]) == 0:
                print("INFO: weights,", weights)
                print("INFO: weights,", np.sum(mapMatrix5)) 
                # mapMatrix5[mapMatrix5 == np.nan] = 0
                continue
            fRobotStates = particle[idx[0][0], :]
            if fRobotStates[2] < 0:
                fRobotStates[2] = fRobotStates[2] + 2 * np.pi
            elif fRobotStates[2] > 2*np.pi:
                fRobotStates[2] = fRobotStates[2] - 2*np.pi

            [score, validScore, curLoss] = self.computeTheLossOnLidar(mapMatrix5, fRobotStates, obstaclesToRobot)

            fStates = fRobotStates # note there need the copy of fRobotStates since fRobotStates will keep changing in the later iteration 
            fStates[:2] = (fStates[:2] - self.mapOffset) / 20.0 # move to the coordinate of UWB map and convert to meter
            if bestLoss > curLoss:
                # varianceNew = self.estimateNewVariance(preBestState, fRobotStates, varianceNew)
                varianceNew[:2] = varianceNew[:2] * 0.8
                varianceNew[2] = varianceNew[2] * 0.8
                bestLoss = curLoss
                updateUwb = True
                preBestState = fStates.copy()
                finalStates = fStates.copy() # note there need the copy of fRobotStates since fRobotStates will keep changing in the later iteration 
                bestScore = score
                bestValidScore = validScore
            iterCnt += 1

        pfState = 1 #Not find the optimal one, donot update the particles and particles are not updated
        # if updateUwb == True: # if the loss from one of the iteration is smaller than the coarsely estimated one, then we will use them to update 
        #     # the old state of the particles, even if we did not find the optimal one.
        #     pfState = 1
        #     if (bestScore < self.threshold and bestValidScore > self.overlapThres): #successfully find the optimal point
        #         pfState = 2
        #         # update the particles
        #         [uwbWeight, headingWeight] = self.evaluateWeightForPos(fStates)
        #         self.weightD = self.weightD * uwbWeight
        #         self.weightD = self.weightD / np.sum(self.weightD)
        #         self.weightH = self.weightH * headingWeight
        #         self.weightH = self.weightH / np.sum(self.weightH)

        # elif (bestScore < self.threshold and bestValidScore > self.overlapThres):
        #     pfState = 2
        #     finalStates = coarsePos.copy()

        print " ------------------- "
        print "bestScore:", bestScore, " threshold:", self.threshold, " bestValidScore:", bestValidScore, " overlapThres:", self.overlapThres
        if (bestScore < self.threshold and bestValidScore > self.overlapThres): #successfully find the optimal point
            pfState = 2
            # update the particles
            [uwbWeight, headingWeight] = self.evaluateWeightForPos(fStates)
            self.weightD = self.weightD * uwbWeight
            self.weightD = self.weightD / np.sum(self.weightD)
            self.weightH = self.weightH * headingWeight
            self.weightH = self.weightH / np.sum(self.weightH)
        else:
            finalStates = coarsePos.copy()
        print "finalStates: ", finalStates
        print "coarsePos: ", coarsePos
        print "posEst: ", posEst
        print "pfState: ", pfState

        self.fStates = finalStates.copy()
        # check if resampling needed
        if 1. / np.sum(self.weightD ** 2) < self.pfCnt / 2.:  # If particle cloud degenerate:
            indx = np.random.choice(self.pfCnt, self.pfCnt, p = self.weightD)
            self.particles[:, :2] = self.particles[indx, :2]# .reshape(self.pfCnt, 4)
            self.weightD = (np.ones((self.pfCnt, 1))*(1.0 / self.pfCnt)).flatten()

        if 1. / np.sum(self.weightH ** 2) < self.pfCnt / 2.:  # If particle cloud degenerate:
            indx = np.random.choice(self.pfCnt, self.pfCnt, p = self.weightH)
            self.particles[:, 2] = self.particles[indx, 2]# .reshape(self.pfCnt, 4)
            self.weightH = (np.ones((self.pfCnt, 1))*(1.0 / self.pfCnt)).flatten()

        # print("INFO: update status, ", pfState, "iterCnt,", iterCnt, "bestLoss:", bestLoss, "score:", score, 'validScore:', validScore)
        # fStates in meter,
        return pfState, finalStates
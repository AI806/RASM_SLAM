import numpy as np
import scipy.signal as signal
import time
class DynamicWindowApproach(object):

    def __init__(self, evalParam, KinematicModel, obstacleR, deltaT, calcTime, stop_side, stop_front, minPassGap):
        self.evalParam = evalParam[:3]
        self.KinematicModel = KinematicModel
        self.obstacleR = obstacleR
        self.deltaT = deltaT
        self.iterCnt = int((calcTime / deltaT) + 1)
        self.minPassGap = minPassGap
        self.c = stop_front
        self.a = - self.c / (stop_side ** 2)

    def setEvalParam(self, evalParam):
        self.evalParam = evalParam[:3]
        # self.obstacleR = obstacleR

    def isMoving(self, obsPt, threshold):
        # dist = np.sqrt( np.sum(obsPt**2, axis=1) )
        obsPt1 = obsPt[np.logical_not(np.isnan(obsPt))]
        var = np.var(obsPt1)
        moving = False

        # print('var:', var, 'ctAnglar:', obsPt, 'obsPt:', obsPt1)
        if var > threshold:
            moving = True
        return moving

    def getObsMeanAngular(self, lidarData, threshold):
        angle = np.arange(0,180).reshape(180,1)
        filter = np.logical_and(lidarData>0.3, lidarData<threshold)
        angle = angle[filter]
        lidar = lidarData[filter]
        obsAngle = np.abs(-np.pi/2 + angle * np.pi / 180)
        thresAngle = (-(lidar - threshold) * (np.pi / 3))/threshold + np.pi / 6
        angles = angle[obsAngle.flatten() < thresAngle.flatten()]
        meanAngle = np.mean(180 - angles)
        return meanAngle

    def stopCases(self, lidarData, angular, offset_x, offset_y):
        # x = a*(y*y) + b*y +c
        stopFlag = False
        if len(lidarData) > 0:
            filt = np.logical_and( lidarData > 0.05, lidarData < 2 ) 
            ranges = lidarData[filt]
            ang = angular[filt]
            obsx = ranges * np.cos(ang) + offset_x
            obsy = ranges * np.sin(ang) + offset_y
            met = self.a * (obsx ** 2) + self.c
            r = obsx < met
            if np.sum(r.flatten()) > 0:
                stopFlag = True

        return stopFlag

    # def stopCases(self, lidarData, curPos, thres):
    #     stopFlag = False
    #     lidarDistToObs = np.sqrt( (lidarData[:, 0] - curPos[0]) ** 2 + (lidarData[:, 1] - curPos[1]) ** 2 ) # in m
    #     minDist = 1000
    #     if len(lidarDistToObs) > 0:
    #         minDist = np.min(lidarDistToObs)
    #         minDist = np.mean(minDist)
    #         if minDist < thres:
    #             idx = np.where(lidarDistToObs == minDist)
    #             idx = idx[0]
    #             if len(idx) > 1:
    #                 idx = idx[0]
    #             obsPos = lidarData[idx, :]
    #             angle = np.abs( np.arctan2( obsPos[0, 1] - curPos[1], obsPos[0, 0] - curPos[0] ) )
    #             thresDist = ( (0.5 - thres) / np.pi) * angle + thres
    #             # print 'thresDist:', thresDist, 'minDist:', minDist, 'angle:', angle * 180 / np.pi, thres
    #             if minDist < thresDist:
    #                 stopFlag = True

    #     return stopFlag, minDist

    # def stopCases(self, lidar, thres):
    #
    #     stopFlag = False
    #     lidarData = lidar[lidar>0.1]
    #     minDist = 1000
    #     if len(lidarData) > 0:
    #         minDist = np.min(lidarData)
    #         if minDist < thres:
    #             idx = np.where(lidarData == minDist)
    #             idx = idx[0]
    #             if len(idx) > 1:
    #                 idx = idx[0]
    #             angle = np.abs(-np.pi/2 + idx * np.pi / 180)
    #             thresAngle = (-(minDist - thres) * (np.pi / 3))/thres + np.pi / 6
    #             if angle < thresAngle:
    #                 stopFlag = True
    #         minDist = np.mean(minDist)
    #     return stopFlag, minDist

    def dwaMethodDocking(self, curPos, refPos, destAngle):
        # print('evalParam', self.evalParam)
        Vs = np.array([0.1, self.KinematicModel[0], -self.KinematicModel[1], self.KinematicModel[1]])
        Vd = np.array([curPos[3]-self.KinematicModel[2]*self.deltaT, curPos[3]+self.KinematicModel[2]*self.deltaT,
                       curPos[4]-self.KinematicModel[3]*self.deltaT, curPos[4]+self.KinematicModel[3]*self.deltaT])

        Vtmp = np.vstack((Vs, Vd))
        Vr = np.array( [np.max(Vtmp[:, 0]), np.min(Vtmp[:, 1]), np.max(Vtmp[:, 2]), np.min(Vtmp[:, 3])] )

        if Vr[1] < Vr[0]:
            Vr[1] = Vr[0] + self.KinematicModel[2] * self.deltaT
        if Vr[3] < Vr[2]:
            Vr[3] = Vr[2] + self.KinematicModel[3] * self.deltaT

        v_m = np.arange(Vr[0], Vr[1], self.KinematicModel[4])
        o_m = np.arange(Vr[2], Vr[3], self.KinematicModel[5])
        cnt = len(v_m) * len(o_m)

        evalDB = np.zeros((cnt, 5))
        trajDB = np.zeros((5 * cnt, self.iterCnt))
        idx = 0
        for vt in v_m:
            for ot in o_m:
                traj = self.GenerateTrajectory(curPos, vt, ot)
                trajDB[idx:idx + 5,:] = traj
                idx = idx + 5

        pathCnt = trajDB.shape[0] / 5
        Xt = np.reshape(trajDB[:, self.iterCnt-1], (pathCnt, 5)).T
        Xt_1 = np.reshape(trajDB[:, self.iterCnt - 2], (pathCnt, 5)).T
        pt_1 = Xt[0:2, :].T
        pt_2 = Xt_1[0:2, :].T
        vector_1 = pt_1 - pt_2
        vector_2 = refPos[:2] - pt_1
        dotProduct = np.sum(vector_1*vector_2, axis=1)
        value = np.sqrt(np.sum(vector_1**2, axis=1)) * np.sqrt(np.sum(vector_2**2, axis=1))
        heading = 180 - np.arccos(dotProduct / value) * 180 / np.pi
        vel = np.abs(Xt[3, :])
        m_dist = np.sqrt( (Xt[0, :] - refPos[0])**2 + (Xt[1, :] - refPos[1])**2 )

        evalDB[:, 0] = Xt[3, :]
        evalDB[:, 1] = Xt[4, :]
        evalDB[:, 2] = heading
        evalDB[:, 3] = m_dist
        evalDB[:, 4] = vel


        evalDB = self.NormalizeEvalDocking(evalDB)
        feval = np.dot(self.evalParam.reshape(1, 3), evalDB[:, 2:].T)
        [indx, indy] = np.where(feval == np.max(feval))
        u = evalDB[indy[0], 0:2]

        return u.reshape(2, 1)

    def dwaMethod(self, curPos, goalPos, obstacles):

        Vs = np.array([0.15, self.KinematicModel[0], -self.KinematicModel[1], self.KinematicModel[1]])
        Vd = np.array([curPos[3]-self.KinematicModel[2]*self.deltaT, curPos[3]+self.KinematicModel[2]*self.deltaT,
                       curPos[4]-self.KinematicModel[3]*self.deltaT, curPos[4]+self.KinematicModel[3]*self.deltaT])

        Vtmp = np.vstack((Vs, Vd))
        Vr = np.array( [np.max(Vtmp[:, 0]), np.min(Vtmp[:, 1]), np.max(Vtmp[:, 2]), np.min(Vtmp[:, 3])] )
        
        if Vr[0] > Vr[1]:
            Vr[0] = Vr[1] - self.KinematicModel[2] * self.deltaT
        if Vr[2] > Vr[3]:
            Vr[2] = Vr[3] - self.KinematicModel[3] * self.deltaT

        [evalDB, trajDB] = self.Evaluation(curPos, Vr, goalPos, obstacles)
        if evalDB is None:
            print('no path to goal!!')
            u = np.array([0, 0])
            return

        evalDB = self.NormalizeEval(evalDB)

        
        feval = np.dot(self.evalParam.reshape(1, 3), evalDB[:, 2:].T)
        # print "--------------------"
        # print "evalDB: ", evalDB
        # print "feval: ", feval
        # print "np.max(feval): ", np.max(feval)
        # print "+++++++++++++++++++"
        [indx, indy] = np.where(feval == np.max(feval))
        u = evalDB[indy[0], 0:2]
        trajDB_planned = trajDB[indy[0]*5:indy[0]*5+5, :]

        return u.reshape(2, 1), trajDB_planned

    def NormalizeEvalDocking(self, EvalDB):

        suma = np.sum(EvalDB[:, 2:], axis=0)
        if suma[0] != 0:
            EvalDB[:, 2] = EvalDB[:, 2] / suma[0]
        if suma[1] != 0:
            EvalDB[:, 3] = np.sign(EvalDB[:, 3]) * (EvalDB[:, 3] / suma[1])
        if suma[2] != 0:
            EvalDB[:, 4] = EvalDB[:, 4] / suma[2]

        return EvalDB

    def NormalizeEval(self, EvalDB):

        suma = np.sum(EvalDB[:, 2:6], axis=0)
        if suma[0] != 0:
            EvalDB[:, 2] = EvalDB[:, 2] / suma[0]
        if suma[1] != 0:
            EvalDB[:, 3] = np.sign(EvalDB[:, 3]) * (EvalDB[:, 3] / suma[1])
        if suma[2] != 0:
            EvalDB[:, 4] = EvalDB[:, 4] / suma[2]
        # if suma[3] != 0:
        #     EvalDB[:, 5] = EvalDB[:, 5] / suma[3]

        return EvalDB

    # def feasibleAreaEva(self, lidarData, Xt, Xt_1, curAngle): #the lidar data is arranged from left to right, 0-180
    def feasibleAreaEva(self, lidarData, vector_1, curAngle, refGoal, curPos): #the lidar data is arranged from left to right, 0-180

        first_zero_idx = 0
        gap_flag = False # whether the range diff between left and right is too big, if it is, then ignore the vacant iterm
        left_bound = False
        min_gap = 1000
        goalTheta = np.arctan2(refGoal[1] - curPos[1], refGoal[0] - curPos[0])
        close_ref_theta = 0

        if goalTheta < 0:
            goalTheta += np.pi * 2
        if goalTheta > np.pi * 2:
            goalTheta -= np.pi * 2
        lidarLen = len(lidarData)
        for idx in np.arange(1, lidarLen):
            if left_bound == False and lidarData[idx] < 30: #find the left edge in lidar data
                first_zero_idx = idx
                left_bound = True
                continue
            if left_bound == True and (lidarData[idx] > 0.3 or idx == lidarLen - 1):#find the right edge in lidar data
                left_bound = False
                leftRange = lidarData[first_zero_idx - 1]
                if leftRange < 30:
                    leftRange = 300.
                rightRange = lidarData[idx]
                if rightRange < 30:
                    rightRange = 300.
                R2 = np.maximum(leftRange, rightRange)
                R1 = np.minimum(leftRange, rightRange)
                theta_gap = (idx - first_zero_idx) * np.pi / 180
                x1 = (R2 - R1) * np.sin(theta_gap / 2)
                x2 = (R2 - R1) * np.cos(theta_gap / 2)
                x3 = 2*R2*np.sin(theta_gap/2) - x1
                dist = np.sqrt(x3**2 + x2**2)
                if dist > self.minPassGap:
                    theta = ((180 - first_zero_idx + 180 - idx) / 2) * np.pi / 180
                    ref_theta = theta + curAngle - np.pi / 2
                    if ref_theta < 0:
                        ref_theta += np.pi * 2
                    if ref_theta > np.pi * 2:
                        ref_theta -= np.pi * 2
                    gap = np.abs(ref_theta - goalTheta)
                    if gap > np.pi:
                        gap = np.pi * 2 - gap
                    if gap < min_gap:
                        min_gap = gap
                        close_ref_theta = ref_theta
                        if R2 < 1.5*R1:
                            gap_flag = True
        if gap_flag == True:
            vector_2 = np.array([np.cos(close_ref_theta), np.sin(close_ref_theta)]).reshape(1, 2)
            dotProduct = np.sum(vector_1 * vector_2, axis=1)
            value = np.sqrt(np.sum(vector_1 ** 2, axis=1)) + 0.001
            direction = 180 - np.arccos(dotProduct / value) * 180 / np.pi
        else:
            direction = 0
        return direction

    def Evaluation_1(self, x, Vr, goal, ob, lidarData, curAngle):

        v_m = np.arange(Vr[0], Vr[1], self.KinematicModel[4])
        o_m = np.arange(Vr[2], Vr[3], self.KinematicModel[5])
        cnt = len(v_m) * len(o_m)

        evalDB = np.zeros((cnt, 6))
        trajDB = np.zeros((5 * cnt, self.iterCnt))
        idx = 0
        for vt in v_m:
            for ot in o_m:
                traj = self.GenerateTrajectory(x, vt, ot)
                trajDB[idx:idx + 5,:] = traj
                idx = idx + 5

        pathCnt = trajDB.shape[0] / 5
        Xt = np.reshape(trajDB[:, self.iterCnt-1], (pathCnt, 5)).T
        Xt_1 = np.reshape(trajDB[:, self.iterCnt - 2], (pathCnt, 5)).T
        pt_1 = Xt[0:2, :].T
        pt_2 = Xt_1[0:2, :].T
        vector_1 = pt_1 - pt_2
        vector_2 = goal - pt_1
        dotProduct = np.sum(vector_1*vector_2, axis=1)
        value = np.sqrt(np.sum(vector_1**2, axis=1)) * np.sqrt(np.sum(vector_2**2, axis=1))
        heading = 180 - np.arccos(dotProduct / value) * 180 / np.pi

        obs = np.sum(ob**2, axis = 1)
        obs = np.tile(obs.T, (pathCnt, 1))
        xxt = np.sum(Xt[0:2, :]**2, axis = 0)
        xxt = np.tile(xxt.T, (ob.shape[0], 1) ).T
        dist = np.sqrt(obs + xxt - 2 * np.dot(Xt[0:2,:].T, ob.T)) - self.obstacleR
        if dist.shape[1] != 0:
            m_dist = np.min(dist, axis=1)
            m_dist = np.minimum(m_dist, 2.)
        else:
            m_dist = 0
        vel = np.abs(Xt[3, :])

        evalDB[:, 0] = Xt[3, :]
        evalDB[:, 1] = Xt[4, :]
        evalDB[:, 2] = heading
        evalDB[:, 3] = m_dist
        evalDB[:, 4] = vel

        return evalDB, trajDB

    def Evaluation(self, x, Vr, goal, ob):

        v_m = np.arange(Vr[0], Vr[1], self.KinematicModel[4])
        o_m = np.arange(Vr[2], Vr[3], self.KinematicModel[5])
        cnt = len(v_m) * len(o_m)

        evalDB = np.zeros((cnt, 5))
        trajDB = np.zeros((5 * cnt, self.iterCnt))
        idx = 0
        for vt in v_m:
            for ot in o_m:
                traj = self.GenerateTrajectory(x, vt, ot)
                trajDB[idx:idx + 5,:] = traj
                idx = idx + 5

        Xt = np.reshape(trajDB[:, -1], (cnt, 5)).T
        theta = Xt[2, :] * 180 / np.pi

        goalTheta = np.arctan2(goal[1] - Xt[1, :], goal[0] - Xt[0, :]) * 180 / np.pi
        heading = 180 - np.abs(goalTheta - theta)

        obs = np.sum(ob**2, axis = 1)
        obs = np.tile(obs.T, (cnt, 1))
        xxt = np.sum(Xt[0:2, :]**2, axis = 0)
        xxt = np.tile(xxt.T, (ob.shape[0], 1) ).T
        dist = np.sqrt(obs + xxt - 2 * np.dot(Xt[0:2,:].T, ob.T)) - self.obstacleR
        if dist.shape[1] != 0:
            m_dist = np.min(dist, axis=1)
            # m_dist = m_dist - np.min(m_dist)
            m_dist = np.minimum(m_dist, 2.)
            # m_dist = np.maximum(m_dist, 4.0)
        else:
            m_dist = 0
        vel = np.abs(Xt[3, :])

        evalDB[:, 0] = Xt[3, :]
        evalDB[:, 1] = Xt[4, :]
        evalDB[:, 2] = heading
        evalDB[:, 3] = m_dist
        evalDB[:, 4] = vel

        return evalDB, trajDB

    def GenerateTrajectory(self, x, vt, ot):

        idx = np.arange(self.iterCnt)
        omega = x[2] + self.deltaT * ot * idx
        # omega[omega > np.pi] = omega[omega > np.pi] - 2*np.pi
        # omega[omega < -np.pi] = omega[omega < -np.pi] + 2 * np.pi
        sin_o = np.sin(omega)
        cos_o = np.cos(omega)
        traj = np.zeros((5, self.iterCnt))
        traj[0,:] = x[0] + np.cumsum(self.deltaT * vt * cos_o)
        traj[1,:] = x[1] + np.cumsum(self.deltaT * vt * sin_o)
        traj[2,:] = omega
        traj[3,:] = vt
        traj[4,:] = ot

        return traj

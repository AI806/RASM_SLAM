import numpy as np
import scipy.signal as signal
import time
from dwa_evaluation import dwa_evaluation
class DynamicWindowApproach(object):

    def __init__(self, evalParam, KinematicModel, obstacleR, deltaT):
        self.evalParam = evalParam
        self.KinematicModel = KinematicModel
        self.obstacleR = obstacleR
        self.deltaT = deltaT
        self.trajLen = int(evalParam[3] / deltaT + 1)

    def dwaMethod(self, X, goalPos, obstacles):

        # Vs = np.array([0.15, self.KinematicModel[0], -self.KinematicModel[1], self.KinematicModel[1]])
        # Vd = np.array([curPos[3]-self.KinematicModel[2]*self.deltaT, curPos[3]+self.KinematicModel[2]*self.deltaT,
        #                curPos[4]-self.KinematicModel[3]*self.deltaT, curPos[4]+self.KinematicModel[3]*self.deltaT])

        # Vtmp = np.vstack((Vs, Vd))
        # Vr = np.array( [np.max(Vtmp[:, 0]), np.min(Vtmp[:, 1]), np.max(Vtmp[:, 2]), np.min(Vtmp[:, 3])] )
        
        Vr = np.zeros(4)
        Vr[0] = np.maximum(0, X[3] - self.KinematicModel[2] * self.deltaT)
        Vr[1] = np.minimum(self.KinematicModel[0], X[3] + self.KinematicModel[2] * self.deltaT)
        Vr[2] = np.maximum(-self.KinematicModel[1], X[4] - self.KinematicModel[3] * self.deltaT)
        Vr[3] = np.minimum(self.KinematicModel[1], X[4] + self.KinematicModel[3] * self.deltaT)

        # print("Vr before: ", Vr)
        # if Vr[1] < Vr[0]:
        #     Vr[0] = Vr[1] - 2 * self.KinematicModel[2] * self.deltaT
        # if Vr[3] < Vr[2]:
        #     Vr[2] = Vr[3] - 2 * self.KinematicModel[3] * self.deltaT

        print("Vr end: ", Vr)

        pssCnt = int( (Vr[1] - Vr[0]) / self.KinematicModel[4] + 1 ) * int( (Vr[3] - Vr[2]) / self.KinematicModel[5] + 1 )

        if pssCnt < 1:
            print('invalid velocity constraint!!')
            u = np.array([0, 0])
            return u, None

        evalDB = np.zeros((pssCnt, 5))
        trajDB = np.zeros((pssCnt*5, self.trajLen))

        obstacles = obstacles.copy(order='C')
        validTraj = dwa_evaluation(evalDB, trajDB, X, Vr, goalPos, obstacles, self.KinematicModel, self.evalParam, self.obstacleR, self.deltaT)

        if validTraj == 0:
            print('no path to goal!!')
            u = np.array([0, 0])
            return u, None

        evalDB = evalDB[:validTraj, :]
        trajDB = trajDB[:validTraj*5, :]
        
        evalDB = self.NormalizeEval(evalDB)
        feval = np.dot(evalDB[:, 2:5], self.evalParam[:3].reshape((3, 1)) )
        idx = np.where( feval == np.max(feval) )

        print("INFO: idx, ", idx)
        u = evalDB[idx[0][0], 0:2]
        trajDB_planned = trajDB[idx[0][0]*5:idx[0][0]*5+5, :]

        return u, trajDB_planned.T

    def NormalizeEval(self, EvalDB): 

        sumDist = np.sum(EvalDB[:, 2]) 
        if sumDist != 0:
            EvalDB[:, 2] = EvalDB[:, 2] / sumDist

        sumHeading = np.sum(EvalDB[:, 3]) 
        if sumHeading != 0:
            EvalDB[:, 3] = EvalDB[:, 3] / sumHeading
        
        sumVel = np.sum(EvalDB[:, 4])
        if sumVel != 0:
            EvalDB[:, 4] = EvalDB[:, 4] / sumVel

        return EvalDB

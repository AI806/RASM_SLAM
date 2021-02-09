import numpy as np
from numpy.linalg import inv

class KFest(object):

    def __init__(self):
        self.stateCnt = 5
        self.F = np.eye(self.stateCnt)
        self.H = np.eye(self.stateCnt)
        self.Q = np.eye(self.stateCnt)
        self.R = np.eye(self.stateCnt)

        self.Q[0, 0] = 0.1
        self.Q[1, 1] = 0.1
        self.Q[2, 2] = (5 * np.pi / 180.0) ** 2

        self.R[0, 0] = 0.1
        self.R[1, 1] = 0.1
        self.R[2, 2] = (3 * np.pi / 180.0) ** 2

        self.dt_mot = 0.1

        self.Xt = np.zeros((self.stateCnt, 1))
        self.Pt = np.eye(self.stateCnt) * 0.1

    def setInitState(self, x):  # x, y, theta, v, omega
        self.Xt[0, 0] = x[0]
        self.Xt[1, 0] = x[1]
        self.Xt[2, 0] = x[2]
        self.Xt[3, 0] = x[3]
        self.Xt[4, 0] = x[4]

    def jacob(self):
        theta = self.dt_mot * self.Xt[4] + self.Xt[2]  # theta + dt * omega
        ssin_dt = np.sin(theta)
        ccos_dt = np.cos(theta)
        ssin = np.sin(self.Xt[2])
        ccos = np.cos(self.Xt[2])
        omega_1 = 1 / (self.Xt[4] + 1e-8)
        vel_omega = self.Xt[3] * omega_1

        self.F[0, 2] = vel_omega * (ccos_dt - ccos)
        self.F[0, 3] = (ssin_dt - ssin) * omega_1
        self.F[0, 4] = -vel_omega * ssin_dt * omega_1 + vel_omega * self.dt_mot * ccos_dt + vel_omega * ssin * omega_1
        self.F[1, 2] = vel_omega * (-ssin + ssin_dt)
        self.F[1, 3] = (ccos - ccos_dt) * omega_1
        self.F[1, 4] = - vel_omega * ccos * omega_1 + vel_omega * ccos_dt * omega_1 + vel_omega * self.dt_mot * ssin_dt
        self.F[2, 4] = self.dt_mot

    def ctrv(self):
        Ut = np.zeros((self.stateCnt, 1))
        Ut[0, 0] = self.Xt[3] * (np.sin(self.Xt[2] + self.Xt[4] * self.dt_mot) - np.sin(self.Xt[2])) / (
                    self.Xt[4] + 1e-8)
        Ut[1, 0] = self.Xt[3] * (np.cos(self.Xt[2]) - np.cos(self.Xt[2] + self.Xt[4] * self.dt_mot)) / (
                    self.Xt[4] + 1e-8)
        Ut[2, 0] = self.Xt[4] * self.dt_mot
        return Ut

    def KF(self, Z):
        Ut = self.ctrv()
        Xt_hat = self.Xt + Ut
        self.jacob()

        Pt = np.dot(np.dot(self.F, self.Pt), self.F.T) + self.Q

        K_tmp = np.dot(np.dot(self.H, Pt), self.H.T)
        K_tmp = inv(K_tmp + self.R)
        K = np.dot(np.dot(Pt, self.H.T), K_tmp)

        self.Xt = Xt_hat + np.dot(K, Z - np.dot(self.H, Xt_hat))
        self.Pt = Pt - np.dot(np.dot(K, self.H), Pt)

    def getState(self):
        return self.Xt



    

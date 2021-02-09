import numpy as np
from dwa_evaluation import dwa_evaluation
import matplotlib.pyplot as plt
import time
import scipy.io as sio
def f(x, u, dt):

    x[0] = x[0] + dt * np.cos(x[2]) * u[0]
    x[1] = x[1] + dt * np.sin(x[2]) * u[0]
    x[2] = x[2] + dt * u[1]
    x[3] = u[0]
    x[4] = u[1]

    return x

def CalcDynamicWindow(x, model, dt):

    Vr = np.zeros(4)
    Vr[0] = np.maximum(0, x[3] - model[2] * dt)
    Vr[1] = np.minimum(model[0], x[3] + model[2] * dt)
    Vr[2] = np.maximum(-model[1], x[4] - model[3] * dt)
    Vr[3] = np.minimum(model[1], x[4] + model[3] * dt)
    # if Vr[1] < Vr[0]:
    #     Vr[1] = Vr[0] + model[2] * dt
    return Vr

def NormalizeEval(EvalDB): 
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


def DynamicWindowApproach(x, model, goal, evalParam, ob, R, dt):

    trajLen = int(evalParam[3] / dt + 1)

    Vr = CalcDynamicWindow(x, model, dt)
    pssCnt = int( (Vr[1] - Vr[0]) / model[4] + 1 ) * int( (Vr[3] - Vr[2]) / model[5] + 1 )
    evalDB = np.zeros((pssCnt, 5))
    trajDB = np.zeros((pssCnt*5, trajLen))
    validTraj = dwa_evaluation(evalDB, trajDB, x, Vr, goal, ob, model, evalParam, R, dt)
    evalDB = evalDB[:validTraj, :]
    trajDB = trajDB[:validTraj*5, :]
    evalDB1 = NormalizeEval(evalDB)

    feval = np.dot( evalDB1[:, 2:5], evalParam[:3].reshape((3, 1)))

    idx = np.where( feval == np.max(feval) )
    u = evalDB1[idx[0][0], 0:2]

    trajDB_planned = trajDB[idx[0][0] * 5:idx[0][0] * 5 + 5, :]

    return u, trajDB, trajDB_planned

def logToProb(prob):
    exp = np.exp(prob)
    return exp / (1 + exp)

def toRadian(degree):
    return degree * np.pi / 180.0

x = np.array([0, 0, np.pi/2.0, 0, 0])
goal = np.array([10.0, 10.0])
obs = np.zeros((200, 200))
obs[30:70, 40:80] = 1
obs[40:60, 140:160] = 1
obs[100:130, 130:150] = 1
obs[90:110, 90:140] = 1
obs[160:180, 80:100] = 1
obs[140:160, 30:60] = 1

[xx, yy] = np.where(obs == 1)
# obstacle =np.array([xx, yy] * 0.05
obstacle = np.vstack((xx, yy)).T
obstacle = obstacle * 0.05
obstacle = obstacle.copy(order='C')
obstacleR = 0.5
dt = 0.1

Kinematic = np.array([0.7, toRadian(70.0), 0.8, toRadian(400.0), 0.02, toRadian(2)])
evalParam = np.array([0.08, 0.2, 0.1, 3.0])
result = None

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)

debuging = False
if debuging == True:
    data = sio.loadmat('/home/guanmy/gazebo_ws/dataRcord/debuging')
    robotState = data['robotState']
    uu = data['u']
    obstacles = data['obstacles']
    refPos = data['refPos']
    trajj = data['traj']

# for i in np.arange(robotState.shape[0]):
for i in np.arange(1, 5000):
    print("-------Step ", i, "---------")
    if debuging == True:
        x = robotState[i, :].copy(order='C')
        goal = refPos[i, :].copy(order='C')
        obstacle = obstacles[:, :, i].reshape(obstacles.shape[0], obstacles.shape[1]).copy(order='C')
        traj1 = trajj[:, :, i].reshape(trajj.shape[0], trajj.shape[1])
        u1 = uu[i, :]
    start_time = time.time()
    [u, traj, trajDB_planned] = DynamicWindowApproach(x, Kinematic, goal, evalParam, obstacle, obstacleR, dt)
    print("Elapsed time: ", time.time() - start_time)

    if debuging == True:

        print("x: ", x)
        print("u: ", u)
        print("u1: ", u1)
    else:
        print("x: ", x)
        print("u: ", u)
        x = f(x, u, dt)

    if result is None:
        result = x.reshape((1, 5))
    else:
        result = np.vstack((result, x))

    ax1.clear()
    plt.plot(result[:, 0], result[:, 1], '-b')
    plt.plot(goal[0], goal[1], '*r')
    plt.plot(obstacle[:, 0], obstacle[:, 1], '.k')


    # for it in np.arange(int(traj.shape[0]/5)):
    #     ind = it * 5
    #     plt.plot(traj[ind, :], traj[ind+1, :], '-g')
    plt.plot(trajDB_planned[0, :], trajDB_planned[1, :], '-r')
    plt.draw()
    plt.pause(0.000001)

# obstacles = np.random.rand(100, 2, 200) * 100
# mapMatrix5 = (np.random.rand(200, 200) - 0.5) * 100
#
# obstaclesTmp = (obstacles + 0.5).astype(int)
# mP5 = mapMatrix5[obstaclesTmp[:, 1, :], obstaclesTmp[:, 0, :]]
# mP5 = logToProb(mP5)
# mP5[np.isnan(mP5)] = 1
#
# objValue = (1 - mP5) ** 2
# objValue = np.mean(objValue, 1)
# idx = np.where(objValue == np.min(objValue))
#
# x = 5000
# print(logToProb(x))
# print(logToProb(-x))
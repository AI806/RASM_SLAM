#include <math.h>
#include <stdlib.h>
void GenerateTrajectory(double* traj, double* Xt, double* Xt_1, double* X, double vt, double wt, double T, double dt);
void f(double* Xt, double vt, double wt, double dt);
double CalcHeadingEval(double* Xt, double* goal);
double CalcDistEval(double* Xt, double* obs, double R, int oSz);
double CalcBreakingDist(double vel, double* model, double dt);
double CalcHeadingEval_bk(double* Xt, double* goal);
#define M_PI 3.14159265358979323846

int c_dwa_evaluation (double* evalDB, double* trajDB, double* X, double* Vr, double* goal, double* obst, double* model, double* evalParam, 
    int oSz, double R, double dt) {

    
    int trajCnt = (int)evalParam[3] / dt + 1;
    double Xt[5];
    double Xt_1[5];
    double traj[trajCnt][5];
    double heading, dist, vel;
    int idxEva = 0;
    int idxTraj = 0;

    int validTraj = 0;
    // printf("trajCnt: %d, oSz: %d\n", trajCnt, oSz);
    // printf("Vr: %f, %f, %f, %f\n", Vr[0], Vr[1], Vr[2], Vr[3]);
    // printf("model: %f, %f, %f, %f, %f, %f\n", model[0], model[1], model[2], model[3], model[4], model[5]);

    int xI, yI;
    int k, l;
    double vt, wt;

    xI = (Vr[1] - Vr[0]) / model[4] + 1;
    yI = (Vr[3] - Vr[2]) / model[5] + 1;

    // printf("xI: %d, yI: %d\n", xI, yI);
    for(k=0; k<xI; k++){
        vt = Vr[0] + k*model[4];
        // printf("vt: %f\n", vt);
        for(l=0; l<yI; l++){
            wt = Vr[2] + l*model[5];
            GenerateTrajectory(&traj[0][0], Xt, Xt_1, X, vt, wt, evalParam[3], dt);
            heading = CalcHeadingEval(Xt, goal);
            dist = CalcDistEval(Xt, obst, R, oSz);  
            vel = fabs(vt);

            double stopDist = CalcBreakingDist(vel, model, dt);
            // printf("stopDist: %f, dist: %f\n", stopDist, dist);
            if (stopDist < dist){
                validTraj ++;

                evalDB[idxEva++] = vt;
                evalDB[idxEva++] = wt;
                evalDB[idxEva++] = heading;
                evalDB[idxEva++] = dist;
                evalDB[idxEva++] = vel;

                for(int i=0; i<5; i++){
                    for(int j=0; j<trajCnt; j++){
                        trajDB[idxTraj++] = traj[j][i];
                    }
                }
                
            }
        }
    }
   
    return validTraj;
}

double CalcBreakingDist(double vel, double* model, double dt){
    double stopDist = 0;
    while(vel >0){
        stopDist += vel*dt;
        vel -= model[2] *dt;
    }
    // printf("dist: %f , vel: %f, acc: %f, dt: %f\n", stopDist, vel, model[2], dt);
    return stopDist;
}

double CalcDistEval(double* Xt, double* obs, double R, int oSz){
    double minDist = 100;
    for (int i=0; i<oSz; i++){
        double dist = sqrt( pow(obs[i*2] - Xt[0], 2) + pow(obs[i*2+1] - Xt[1], 2) ) - R;
        if (minDist > dist)
            minDist = dist;
    }
    // printf("minDist: %f, Xt: %f, %f \n", minDist, Xt[0], Xt[1]);
    if (minDist >= 2*R)
        minDist = 2*R;

    return minDist;
}

double CalcHeadingEval_bk(double* Xt, double* goal){
    double theta = Xt[2] * 180.0 / M_PI;
    double goalTheta;

    if (theta > M_PI){
        theta = theta - 2 * M_PI;
    } else if (theta < -M_PI){
        theta = theta + 2 * M_PI;
    }

    goalTheta = atan2(goal[1] - Xt[1], goal[0] - Xt[0]);
    goalTheta = goalTheta * 180.0 / M_PI;

    // printf("goalTheta: %f, theta: %f, goal: %f, %f, Xt: %f, %f\n", goalTheta, theta, goal[0], goal[1], Xt[0], Xt[1]);
    
    double diff = fabs(goalTheta - theta);

    if (diff > M_PI){
        diff = - diff + 2 * M_PI;
    }
    // printf("diff: %f\n", diff);
    return 180.0 - diff;
}

double CalcHeadingEval(double* Xt, double* goal){

    double theta = Xt[2];
    double v1_1 = cos(theta);
    double v1_2 = sin(theta);
    double v2_1 = goal[0] - Xt[0];
    double v2_2 = goal[1] - Xt[1];
    double d = sqrt( pow(v2_1, 2) + pow(v2_2, 2) );

    v2_1 = v2_1 / d;
    v2_2 = v2_2 / d;

    double dotSum = v1_1 * v2_1 + v1_2 * v2_2;
    double heading = acos(dotSum) * 180 / M_PI;

    // printf("v1: %f, %f, v2: %f, %f, dotSum: %f, dSum: %f\n", v1_1, v1_2, v2_1, v2_2, dotSum, divideSum);
    
    return 180.0 - heading;
}

void GenerateTrajectory(double* traj, double* Xt, double* Xt_1, double* X, double vt, double wt, double T, double dt) {
    double time = 0;

    int idx = 0;
    for(int i=0; i<5; i++){
        traj[idx++] = X[i];
        Xt[i] = X[i];
    }
    while(time <= T){
        time += dt;
        for(int i=0; i<5; i++){
            Xt_1[i] = Xt[i];
        }
        f(Xt, vt, wt, dt);
        for(int i=0; i<5; i++){
            traj[idx++] = Xt[i];
        }
    }
}

void f(double* Xt, double vt, double wt, double dt){
    Xt[0] = Xt[0] + dt * cos(Xt[2]) * vt;
    Xt[1] = Xt[1] + dt * sin(Xt[2]) * vt;
    Xt[2] = Xt[2] + dt * wt;
    Xt[3] = vt;
    Xt[4] = wt;
}
#include <vector>
#ifndef RASM_TOOLS_H
#define RASM_TOOLS_H
#define STATECNT 3

struct CellData
{
    int i_, j_;
    int src_i, src_j;
};


class RASM_tools {

    private:
        double resol, laser_sigma_hit, laser_likelihood_max_dist, laser_z_rand, range_max, laser_z_hit;
        int pRefineCnt, maxIter, maxBeam, mapSz_x, mapSz_y, succCnt, checkTimes;
        int maxRefCnt, minRefCnt;
        int cell_radius_, totalMapSz;
        double mapOffset_x, mapOffset_y;

        // stop cretia
        double closeOptimalLoss, stopLoss, optimal_stop_loss;

        double **distances_;
        unsigned char *marked;
        double *max_occ_dist;

        int status;
        double refine_score, bestCaseScore, worseCaseScore;

        double refineStd[STATECNT], maxStd[STATECNT], minStd[STATECNT];
        double bestPos[STATECNT];
        double *preComputeScore;
        double bestRatio;

        double gauss(double mu, double std);
        
        void enqueue(int i, int j, int src_i, int src_j, std::vector<CellData> &Q);
        double findBestParticle(double *obstaclesToRobot, double *var, int obsCnt);
        int updateVariance(double lT1, double lT2, double lT3, double *preSigma);
        double logToProb(double prob);
        double scoreForRobotState(double *robotState, double *map, double *obstaclesToRobot, int obsCnt);

    public:

        RASM_tools();
        ~RASM_tools();

        void initProcess();

        void setLaserParam(double l_sigma_hit, double l_likelihood_max_dist, double l_z_rand, double r_max, double l_z_hit, int m_beam);
        void setMapParam(int mSize_x, int mSize_y, double mOffset_x, double mOffset_y, double r);
        void setpfParam(double *rStd, double *maStd, double *miStd, int pCnt, int mIter, double cOptimalLoss, double sLoss, 
            double o_stop_loss, int cTimes, int maRefCnt, int miRefCnt);

        void getOptimalState(double *state, int *status, double *score);
        void getOccMap(double *occMap);
        void computeOccMap(double *map);

        double refineRobotState(double *map, double *robotState, double *obstaclesToRobot, int obsCnt);
        void freeMem();
};


#endif
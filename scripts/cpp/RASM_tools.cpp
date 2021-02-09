#include <iostream>
#include <math.h>
#include "RASM_tools.h"

#define MAP_INDEX(mapSz, i, j) ((i) + (j) * mapSz)
#define MAP_CHECK(mapSz, x) ( x>=mapSz ? mapSz-1 : (x<0 ? 0 : x) )
#define MAP_VALID(mapS_x, mapS_y, i, j) ((i >= 0) && (i < mapS_x) && (j >= 0) && (j < mapS_y))


// Default constructor
RASM_tools::RASM_tools () {}

// Destructor
RASM_tools::~RASM_tools () {
    
}

void RASM_tools::freeMem(){
    if(distances_)
    {
        for(int i=0; i<=cell_radius_+1; i++)
            delete[] distances_[i];
        delete[] distances_;
    }
    delete[] marked;
    delete[] max_occ_dist;
    delete[] preComputeScore;
}

double RASM_tools::gauss(double mu, double std)
{
    double x = (double)random() / RAND_MAX,
        y = (double)random() / RAND_MAX,
        z = sqrt(-2 * log(x)) * cos(2 * M_PI * y);
    return mu + z * std;
}

double RASM_tools::logToProb(double prob){
    
    if (prob > 10){
        return 1;
    } else {
        double expVal = exp(prob);
        return expVal / (1 + expVal);
    }

}
        
void RASM_tools::setLaserParam(double l_sigma_hit, double l_likelihood_max_dist, double l_z_rand, double r_max, double l_z_hit,
        int m_beam){
    laser_sigma_hit = l_sigma_hit;
    laser_likelihood_max_dist = l_likelihood_max_dist;
    laser_z_rand = l_z_rand;
    range_max = r_max;
    laser_z_hit = l_z_hit;
    maxBeam = m_beam;

    bestCaseScore = pow((laser_z_hit + laser_z_rand / range_max), 3);
    worseCaseScore = pow( (laser_z_hit * exp(-(laser_likelihood_max_dist*laser_likelihood_max_dist)/(2.0 * laser_sigma_hit * laser_sigma_hit)) 
         + laser_z_rand / range_max), 3);

    std::cout << "laser param, l_sigma_hit: " << l_sigma_hit << ", l_likelihood_max_dist: " << l_likelihood_max_dist << ", l_z_rand: " << l_z_rand <<
            ", r_max: " << r_max << ", l_z_hit: " << l_z_hit << ", m_beam: " << m_beam << std::endl;
}
void RASM_tools::setMapParam(int mSize_x, int mSize_y, double mOffset_x, double mOffset_y, double r){
    mapOffset_x = mOffset_x;
    mapOffset_y = mOffset_y;
    mapSz_x = mSize_x;
    mapSz_y = mSize_y;
    resol = r;

    std::cout << "map param, mSize_x: " << mSize_x << ", mapSz_y: " << mapSz_y << ", mOffset_x: " << mOffset_x << 
        ", mOffset_y: " << mOffset_y << ", r: " << r << std::endl;
}
void RASM_tools::setpfParam(double *rStd, double *maStd, double *miStd, int pCnt, int mIter, double cOptimalLoss, 
    double sLoss, double o_stop_loss, int cTimes, int maRefCnt, int miRefCnt){
    pRefineCnt = pCnt;
    maxRefCnt = maRefCnt;
    minRefCnt = miRefCnt;
    maxIter = mIter;
    closeOptimalLoss = cOptimalLoss;
    stopLoss = sLoss;
    checkTimes = cTimes;
    optimal_stop_loss = o_stop_loss;

    for (int i=0; i<STATECNT; i++){
        refineStd[i] = rStd[i];
        maxStd[i] = maStd[i];
        minStd[i] = miStd[i];
    }
    
    std::cout << "pf param, rStd: " << rStd[0] << " " << rStd[1] << " " << rStd[2] << ", maStd: " << maStd[0] << " " << maStd[1] << " " << maStd[2] <<
            ", miStd: " << miStd[0] << " " << miStd[1] << " " << miStd[2] << ", pCnt: " << pCnt << ", mIter: " << mIter << ", cOptimalLoss: " << 
            cOptimalLoss << ", sLoss: " << sLoss << ", optimal_stop_loss: " << optimal_stop_loss << ", cTimes:" << cTimes << std::endl;

}

void RASM_tools::initProcess(){
    
    cell_radius_ = static_cast<int>(laser_likelihood_max_dist / resol + 0.5);
    distances_ = new double *[cell_radius_+2];
    for(int i=0; i<=cell_radius_+1; i++)
    {
        distances_[i] = new double[cell_radius_+2];
        for(int j=0; j<=cell_radius_+1; j++)
        {
            distances_[i][j] = sqrt(i*i + j*j);
        }
    }

    int cnt = static_cast<int>(laser_likelihood_max_dist / 0.01);
    preComputeScore = new double[cnt+2];
    double z_hit_denom = 2 * laser_sigma_hit * laser_sigma_hit;
    double z_rand_mult = laser_z_rand / range_max;
    for(int i=0; i<cnt+2; i++){
        double z = i*0.01;
        double pz = laser_z_hit * exp(-(z * z) / z_hit_denom) + z_rand_mult;
        preComputeScore[i] = pz*pz*pz;
    }

    totalMapSz = mapSz_x*mapSz_y;

    marked = new unsigned char[totalMapSz];
    max_occ_dist = new double[totalMapSz];

    std::fill(marked, marked + totalMapSz, 0);
    std::fill(max_occ_dist, max_occ_dist + totalMapSz, laser_likelihood_max_dist);
}

void RASM_tools::enqueue(int i, int j, int src_i, int src_j, std::vector<CellData> &Q){
        
    int mapIdx = MAP_INDEX(mapSz_y, i, j);
    if (marked[mapIdx] == 1)
        return;
    int di = abs(i - src_i);
    int dj = abs(j - src_j);
    double occ_dist = distances_[di][dj];

    if (occ_dist >= cell_radius_)
        return;

    CellData cell;
    cell.src_i = src_i;
    cell.i_ = i;
    cell.src_j = src_j;
    cell.j_ = j;
    Q.push_back(cell);   

    marked[mapIdx] = 1;
    max_occ_dist[mapIdx] = occ_dist * resol;
}

void RASM_tools::computeOccMap(double *map){

    // initiall the global martix, include the marker flag and max_occ_dist
    std::fill(marked, marked + totalMapSz, 0);
    std::fill(max_occ_dist, max_occ_dist + totalMapSz, laser_likelihood_max_dist);

    std::vector<CellData> cells;

    for (int i=0; i<mapSz_y; i++){
        for (int j=0; j<mapSz_x; j++){
            int mapIdx = MAP_INDEX(mapSz_y, i, j);
            // if (logToProb(map[mapIdx]) > 0.7){
            if ((map[mapIdx]) > 0.8){
                max_occ_dist[mapIdx] = 0.0;
                marked[mapIdx] = 1;
                CellData cell;
                cell.src_i = cell.i_ = i;
                cell.src_j = cell.j_ = j;
                cells.push_back(cell);
            }  
        }
    }
    int cellIdx = 0;
    while (cellIdx < cells.size()){
        CellData current_cell = cells[cellIdx];
        if (current_cell.i_ > 0){
            enqueue(current_cell.i_ - 1, current_cell.j_, current_cell.src_i, current_cell.src_j, cells);
        }
        if (current_cell.j_ > 0){
            enqueue(current_cell.i_, current_cell.j_ - 1, current_cell.src_i, current_cell.src_j, cells);
        }
        if (current_cell.i_ < mapSz_y - 1){
            enqueue(current_cell.i_ + 1, current_cell.j_, current_cell.src_i, current_cell.src_j, cells);
        }
        if (current_cell.j_ < mapSz_x - 1){
            enqueue(current_cell.i_, current_cell.j_ + 1, current_cell.src_i, current_cell.src_j, cells);
        }
        // cells.erase(cells.begin());
        cellIdx ++;
    }
    cells.clear();
}

double RASM_tools::findBestParticle(double *obstaclesToRobot, double *var, int obsCnt)
{
    int step = static_cast<int>((obsCnt - 1) / (static_cast<double>(maxBeam) - 1.0));

    if (step < 1)
        step = 1;

    for (int i=0; i<pRefineCnt; i++){
        double base_x, base_y, base_theta;
        base_x = gauss(bestPos[0], var[0]) / resol + mapOffset_x;
        base_y = gauss(bestPos[1], var[1]) / resol + mapOffset_y;
        base_theta = gauss(bestPos[2], var[2]);

        double ccos = cos(base_theta);
        double ssin = sin(base_theta);
        double p = 0;
        double succNum = 0, iterCnt = 0.0;
        for (int j=0; j<obsCnt; j+=step){
            double pz = 0.0, z = laser_likelihood_max_dist;
            double x = obstaclesToRobot[j*2], y = obstaclesToRobot[j*2+1];
            int robot_x = static_cast<int>(lround(ccos * x - ssin * y + base_x));
            int robot_y = static_cast<int>(lround(ssin * x + ccos * y + base_y));

            if (MAP_VALID(mapSz_y, mapSz_x, robot_x, robot_y)){
                z = max_occ_dist[MAP_INDEX(mapSz_y, robot_x, robot_y)];
            }
            if (z < 1e-5){
                p += bestCaseScore;
                iterCnt ++;
            } else if (z >= laser_likelihood_max_dist) { 
                p += worseCaseScore;
            } else {
                int idx = static_cast<int>(z / 0.01);
                p += preComputeScore[idx];
            }
            succNum ++;
        }
        p = p / succNum;
        if (p > refine_score){
            refine_score = p;
            bestRatio = iterCnt / succNum;
            bestPos[0] = (base_x - mapOffset_x) * resol;
            bestPos[1] = (base_y - mapOffset_y) * resol;
            bestPos[2] = base_theta;
        }
    }
    return refine_score;
}

int RASM_tools::updateVariance(double lT1, double lT2, double lT3, double *preSigma){
    
    double rho = (lT2 - lT3) / (lT1 - lT2 + 1e-5);

    if (rho < 1e-3 && lT1 < closeOptimalLoss)
        rho = 1.5;

    if ( (lT1 - lT2) < 1e-5 && lT1 > stopLoss ){
        succCnt += 1;
        if (succCnt > checkTimes)
            return 1;
    }  
    else
        succCnt = 0;

    if (rho > 5)
        rho = 5;
    if (rho < 0.2)
        rho = 0.2;

    preSigma[0] = rho * preSigma[0];
    preSigma[1] = rho * preSigma[1];
    preSigma[2] = rho * preSigma[2];

    if (preSigma[0] > maxStd[0])
        preSigma[0] = maxStd[0];
    else if (preSigma[0] < minStd[0])
        preSigma[0] = minStd[0];

    if (preSigma[1] > maxStd[1])
        preSigma[1] = maxStd[1];
    else if (preSigma[1] < minStd[1])
        preSigma[1] = minStd[1];

    if (preSigma[2] > maxStd[2])
        preSigma[2] = maxStd[2];
    else if (preSigma[2] < minStd[2])
        preSigma[2] = minStd[2];

    // pRefineCnt = pRefineCnt * rho;
    // if (pRefineCnt > maxRefCnt)
    //     pRefineCnt = maxRefCnt;
    // if (pRefineCnt < minRefCnt)
    //     pRefineCnt = minRefCnt;

    return 0;
}

double RASM_tools::scoreForRobotState(double *robotState, double *map, double *obstaclesToRobot, int obsCnt){
    double base_x, base_y, base_theta;

    base_x = robotState[0] / resol + mapOffset_x;
    base_y = robotState[1] / resol + mapOffset_y;
    base_theta = robotState[2];

    double ccos = cos(base_theta);
    double ssin = sin(base_theta);
    double tMapValue = 0, succNum = 0.0;
    for (int j=0; j<obsCnt; j++){
        double mapValue = 0;
        double x = obstaclesToRobot[j*2], y = obstaclesToRobot[j*2+1];
        int robot_x = static_cast<int>(lround(ccos * x - ssin * y + base_x));
        int robot_y = static_cast<int>(lround(ssin * x + ccos * y + base_y));

        if (MAP_VALID(mapSz_y, mapSz_x, robot_x, robot_y)){
            if (map[MAP_INDEX(mapSz_y, robot_x, robot_y)] > 0.8)
                mapValue = 1;
        }
        tMapValue += pow((1 - mapValue), 2);
        succNum ++;
    }
    tMapValue = tMapValue / succNum;
    return tMapValue;
}

double RASM_tools::refineRobotState(double *map, double *robotState, double *obstaclesToRobot, int obsCnt){

    int iterCnt = 0;
    refine_score = -1;
    bestRatio = 0;
    succCnt = 0;
    // copy the state for the variance that used to refine the robot's state 
    // set the initial state of the robot
    double varianceNew[STATECNT];
    for( int i=0; i<STATECNT; i++){
        varianceNew[i] = refineStd[i];
        bestPos[i] = robotState[i];
    }
    
    double lT1 = -1, lT2 = -1, lT3 = -1;
    int endFlag = 0;
    while (iterCnt < maxIter && endFlag == 0){
        
        // find the best particle among current particles
        refine_score = findBestParticle(obstaclesToRobot, varianceNew, obsCnt);
        if (refine_score == -1){
            printf("Encounter unknown error!");
            status = 0;
            return 0;
        }
        if (iterCnt == 0)
            lT3 = refine_score;
        else if (iterCnt == 1)
            lT2 = refine_score;
        else{
            lT1 = refine_score;
            endFlag = updateVariance(lT1, lT2, lT3, varianceNew);
            lT2 = lT1; lT3 = lT2;
        }
        iterCnt ++;
    }

    if ( (refine_score > optimal_stop_loss) || ( bestRatio > 0.6 ))
        status = 1;
    else
    {
        status = 0;
    }
    return iterCnt;
}

void RASM_tools::getOptimalState(double *state, int *sta, double *sco){
    
    if (status == 1){
        for (int i=0; i<STATECNT; i++){
            state[i] = bestPos[i];
        }
    }
    *sco = refine_score;
    *sta = status;
}
void RASM_tools::getOccMap(double *occMap){
    memcpy(occMap, max_occ_dist, sizeof(double)*mapSz_x*mapSz_y);
}
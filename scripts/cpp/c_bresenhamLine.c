#include <math.h>

int c_bresenhamLine (double* mapMatrix, int* mapMatrixIndex, int* startPos, int* endPos, int xSize,
    int ySize, int pointCnt, double updateFree, double updateOcc, int currMarkFreeIndex, int currMarkOccIndex) {

    int i, j;
    int index = 0;
    int startx, starty, endx, endy;
    int tmpEndPosIndex = 0;
    startx = startPos[0];
    starty = startPos[1];
    if (startx < 0 || startx >= xSize){
        return 0;
    }
    if (starty < 0 || starty >= ySize){
        return 0;
    }
    for (i = 0; i < pointCnt; i++) {

        endx = endPos[tmpEndPosIndex];
        endy = endPos[tmpEndPosIndex + 1];
        tmpEndPosIndex += 2;

        if ((endx < 0) || (endx >= xSize) || (endy < 0) || (endy >= ySize)) {
            continue;
        }
        if ((startx != endx) && (starty != endy)){
            updateLineBresenhami(mapMatrix, mapMatrixIndex, startx, starty, endx, endy, xSize, updateFree, updateOcc, currMarkFreeIndex, currMarkOccIndex);
        }
    }
    return 1;
}

void updateLineBresenhami(double* mapMatrix, int* mapMatrixIndex, int x0, int y0, int x1, int y1, int xSize, double updateFree, double updateOcc, int currMarkFreeIndex, int currMarkOccIndex){

    int error_y, dx, dy, offset_dx, offset_dy, error_x;
    unsigned int abs_dx, abs_dy, startOffset, endOffset;

    dx = x1 - x0;
    dy = y1 - y0;

    abs_dx = abs(dx);
    abs_dy = abs(dy);

    offset_dx = dx > 0 ? 1 : -1;
    offset_dy = (dy > 0 ? 1 : -1) * xSize;

    startOffset = y0 * xSize + x0;

    if(abs_dx >= abs_dy){
        error_y = abs_dx / 2;
        bresenham2D(mapMatrix, mapMatrixIndex, abs_dx, abs_dy, error_y, offset_dx, offset_dy, startOffset, updateFree, currMarkFreeIndex);
    } else {
        error_x = abs_dy / 2;
        bresenham2D(mapMatrix, mapMatrixIndex, abs_dy, abs_dx, error_x, offset_dy, offset_dx, startOffset, updateFree, currMarkFreeIndex);
    }
    endOffset = y1 * xSize + x1;

    if (mapMatrixIndex[endOffset] < currMarkOccIndex){
        if (mapMatrixIndex[endOffset] == currMarkFreeIndex){
            mapMatrix[endOffset] -= updateFree;
        }
        mapMatrix[endOffset] += updateOcc;
        mapMatrixIndex[endOffset] = currMarkOccIndex;
    }
}

void bresenham2D(double* mapMatrix, int* mapMatrixIndex, unsigned int abs_da, unsigned int abs_db, int error_b, int offset_a, int offset_b, unsigned int offset, double updateFree, int currMarkFreeIndex){

    unsigned int end, i;
    end = abs_da - 1;
    if (mapMatrixIndex[offset] < currMarkFreeIndex){
        mapMatrix[offset] += updateFree;
        mapMatrixIndex[offset] = currMarkFreeIndex;
    }

    for(i = 0; i < end; ++i)
    {
        offset += offset_a;
        error_b += abs_db;
        if((unsigned int)error_b >= abs_da)
        {
            offset += offset_b;
            error_b -= abs_da;
        }
        if (mapMatrixIndex[offset] < currMarkFreeIndex){
            mapMatrix[offset] += updateFree;
            mapMatrixIndex[offset] = currMarkFreeIndex;
        }
    }
}
import numpy as np
import re
import sys
# def isOcclusioned(maps, p1, p2):
#
#     minx = np.min([p2[0], p1[0]])
#     miny = np.min([p2[1], p1[1]])
#     maxx = np.max([p2[0], p1[0]])
#     maxy = np.max([p2[1], p1[1]])
#     maxDist = np.max([maxy - miny, maxx - minx])
#
#     point_set = np.zeros([2, maxDist], dtype=np.int)
#     point_set[0] = np.linspace(minx, maxx, maxDist)
#     point_set[1] = np.linspace(miny, maxy, maxDist)
#
#     pixels = maps[[point_set[0,],point_set[1,]]]
#     if pixels.sum() > 0:
#         collision =  True
#     else:
#         collision = False
#     return collision

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

def constructMap():
    # const_maps = np.zeros((30, 65))
    # const_maps[0:2, 0:65] = 1
    # const_maps[0:6, 49:65] = 1
    # const_maps[22:30, 0:65] = 1
    # const_maps = np.zeros((300, 650))
    # const_maps[0:20, 0:650] = 1
    # const_maps[0:60, 490:650] = 1
    # const_maps[220:300, 0:650] = 1

    const_maps = np.zeros((1300, 950))
    const_maps[0:20, 0:695] = 1
    const_maps[0:62, 696:950] = 1
    const_maps[738:1300, 0:395] = 1
    const_maps[970:1300, 893:950] = 1
    # const_maps[62:970, 937:950] = 1
    return const_maps

def isOcclusioned(maps, p1, p2, extend, minDist):
    mapSz = maps.shape
    start = np.zeros((1,2))
    goal = np.zeros((1,2))
    rate = 0.25
    if p1[0] < p2[0]:
        start[0, 0] = p1[0] + (p2[0] - p1[0]) * rate
        goal[0, 0] = p2[0] - (p2[0] - p1[0]) * rate
    else:
        start[0, 0] = p1[0] - (p1[0] - p2[0]) * rate
        goal[0, 0] = p2[0] + (p1[0] - p2[0]) * rate

    if p1[1] < p2[1]:
        start[0, 1] = p1[1] + (p2[1] - p1[1]) * rate
        goal[0, 1] = p2[1] - (p2[1] - p1[1]) * rate
    else:
        start[0, 1] = p1[1] - (p1[1] - p2[1]) * rate
        goal[0, 1] = p2[1] + (p1[1] - p2[1]) * rate

    if goal[0, 0] != start[0, 0]:
        theta = np.arctan(np.abs( (goal[0, 1]-start[0, 1])/(goal[0, 0]-start[0, 0]) ))
    else:
        theta = np.pi/2
    offX = extend * np.sin(theta)
    offY = extend * np.cos(theta)
    polyx = np.array([start[0, 0]-offX, goal[0, 0]-offX, goal[0, 0]+offX, start[0, 0]+offX, start[0, 0]-offX]).astype(int)
    polyy = np.array([start[0, 1]+offY, goal[0, 1]+offY, goal[0, 1]-offY, start[0, 1]-offY, start[0, 1]+offY]).astype(int)

    lu = np.array([np.min(polyx), np.min(polyy)])
    rb = np.array([np.max(polyx), np.max(polyy)])
    lu[lu<0] = 0
    if rb[0] > mapSz[0]:
        rb[0] = mapSz[0]
    if rb[1] > mapSz[1]:
        rb[1] = mapSz[1]
    outerMap = maps[lu[1]:rb[1], lu[0]:rb[0]]
    [pointX, pointY] = np.where(outerMap == 1)

    hasObsInter = False
    if len(pointX) >= 1:
        p1 = start[0,:] - lu + 1
        p2 = goal[0,:] - lu + 1
        v1 = p2[0]-p1[0]
        v2 = p2[1]-p1[1]
        if v1 == 0:
            distMat = np.abs(pointY - p1[0])
        elif v2 == 0:
            distMat = np.abs(pointX - p1[1])
        else:
            A = v2
            B = -v1
            C = v1*p1[1] - v2*p1[0]
            distMat = np.abs( (A*pointY+B*pointX+C) / np.sqrt(A**2+B**2) )
        filter = distMat < minDist
        if np.sum(filter) > 0:
            hasObsInter = True
    return hasObsInter

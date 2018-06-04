#coding=utf-8
import numpy as np

def IsInROI(x, y, ROI):
    
    return (x > ROI[0]) and (x < ROI[0] + ROI[2]) and \
           (y > ROI[1]) and (y < ROI[1] + ROI[3])
def nestedROIs(roi1, roi2):
    
    x1 = roi1[0]
    y1 = roi1[1]
    w1 = roi1[2]
    h1 = roi1[3]

    x2 = roi2[0]
    y2 = roi2[1]
    w2 = roi2[2]
    h2 = roi2[3]

    return (x1 < x2 < x1 + w1) and (y1 < y2 < y1 + h1) or \
           (x2 < x1 < x2 + w2) and (y2 < y1 < y2 + h2)
def Match_Features(set0, set1, minDistThr, minAreaThr):
    list0 = []
    list1 = []

    for a, b, w1, h1 in set0:
        for c, d, w0, h0 in set1:

            EucDist = ((a + w1 / 2) - (c + w0 / 2)) ** 2 - ((b + h1 / 2) - (d + h0 / 2)) ** 2
            AreaDist = np.abs(w0 * h0 - w1 * h1)

            if EucDist > 0:
                EucDist = np.sqrt(EucDist)

            if EucDist < minDistThr and AreaDist < minAreaThr:

                if not [a, b, w1, h1] in list0:
                    list0.append([a, b, w1, h1])

                if not [c, d, w0, h0] in list1:
                    list1.append([c, d, w0, h0])

    set0_out = np.array(list0)
    set1_out = np.array(list1)

    return set0_out, set1_out

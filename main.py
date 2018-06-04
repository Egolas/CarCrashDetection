# coding=utf-8
import cv2
import sys
import os
import getopt
from track import track_params
import track
from time import time


class struct:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def main(argv):
    test_video = '/home/aythior/文档/traffic_accident/Inputs/Sample10.mp4'
    args = struct(
        in_video=test_video,
        params=track_params(),
        scale=2,
        inter_frames_alerts=90
    )
    try:
        opts, user_args = getopt.getopt(argv, "f:", ["params", "scale",
                                                     "inter_frames_alerts"])
        for opt, user_args in opts:
            if opt in "-f":
                args.in_video = user_args
            elif opt in "--params":
                args.params = user_args
            elif opt in "--scale":
                args.scale = user_args
            elif opt in "inter_frames_alert":
                args.inter_frames_alert = user_args
    except getopt.GetoptError:
        sys.exit(2)

    if args.scale == 2:
        args.params['LKT']['lk_params'] = dict(winSize=(5, 5),
                                               maxLevel=3,
                                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        args.params['VJ']['cascade_params'] = dict(scaleFactor=1.01,
                                                   minNeighbors=2,
                                                   maxSize=(300, 300),
                                                   minSize=(5, 5))
    res = {"alerts": {}}

    res = track.Traffic_Tracker(args.in_video,
                                args.params,
                                args.scale,
                                args.inter_frames_alerts
                                )

    print ""
    for i, (key, value) in enumerate(res['alerts'].iteritems(), 1):
        a = value['alX']
        b = value['alY']
        w_ = 70
        h_ = 70
        print ('Frame #%d , alertX: %d, alertY: %d , Alerts=%d/%d' %
               (value['alTime'], a, b, i, len(res['alerts'])))

    print (args.in_video + 'Done!')


if __name__ == "__main__":
    t0 = time()
    print 'Start'

    main(sys.argv[1:])
    t1 = time()
    timeused = t1 - t0

    print('Done! (%.2f sec)' % timeused)

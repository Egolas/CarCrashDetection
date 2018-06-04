# coding=utf-8
import cv2
# import sys
# import os
import numpy as np
from sys import platform
from ROIandFeatures import IsInROI
from ROIandFeatures import nestedROIs
from ROIandFeatures import Match_Features
from vehicleDetector import VehicleDetector

# from subprocess import call
# from datetime import datetime

def track_params():
    params = {'VJ': {},
              'LKT': {},
              'Caffe': {}}
    params['LKT']['max_points'] = 50  # 最大感兴趣点数
    params['LKT']['min_points'] = 10  # 最小感兴趣点数
    params['LKT']['velocity_thr'] = 1  # 速度阈值 (pixels-movement over 2 frames)

    # shi-Tomasi角点检测参数:
    params['LKT']['feature_params'] = dict(maxCorners=params['LKT']['max_points'],  # 图像的角点
                                           qualityLevel=0.01,  # 最小可接受的角点质量
                                           minDistance=5,  # 角点之间最小欧式距离
                                           blockSize=3)  # 计算离散卷积块的大小
    # Lucas-Kanade光流参数
    params['LKT']['lk_params'] = dict(winSize=(15, 15),  # 搜索窗
                                      maxLevel=3,  # 最大金字塔层数
                                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    params['VJ']['velocity_thr'] = 10  # 速度阈值 (pixels-movement over 2 frames)
    params['VJ']['area_thr'] = 1000  # 区域参数(pixels^2),

    params['VJ']['VJ_Classifier'] = 'cars1.xml'

    params['VJ']['cascade_params'] = dict(scaleFactor=1.01,  # 放缩参数
                                          minNeighbors=2,  # 候选矩阵的邻居数
                                          maxSize=(200, 200),  # 最大目标尺寸
                                          minSize=(50, 50))  # 最小目标尺寸
    params['Caffe']['paths'] = dict(caffe_root='/home/aythior/install_env/caffe-master',
                                    caffe_model='./models/VGGNet/vehicle/SSD_300x300/VGG_vehicle_SSD_300x300_iter_31971.caffemodel',
                                    deploy_file='./models/VGGNet/vehicle/SSD_300x300/deploy.prototxt',
                                    labels_file='./data/vehicle/labelmap_vehicle.prototxt')

    return params


def Traffic_Tracker(src, params, scale, inter_frames_alerts):
    caffe_root = params['Caffe']['paths']['caffe_root']
    caffemodel = params['Caffe']['paths']['caffe_model']
    deploy = params['Caffe']['paths']['deploy_file']
    labels_file = params['Caffe']['paths']['labels_file']

    vehicleDetector = VehicleDetector(caffe_root, caffemodel, deploy, labels_file)

    res = {'alerts': {}}
    line_type = cv2.CV_AA
    # 一 数据的输入
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Error loading source file: " + src)
        exit(1)
    w, h = 0, 0
    if platform == 'linux2':
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        w = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH) * scale)
        h = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) * scale)
        framesNum = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    else:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        w = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH) * scale)
        h = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) * scale)
        framesNum = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    # 设定fourcc参数
    fourcc = []
    if platform == 'darwin':
        fourcc = cv2.VideoWriter_fourcc('S', 'V', 'Q', '3')  # MacOs
    elif platform == 'linux2':
        fourcc = cv2.cv.CV_FOURCC(*'XVID')  # Linux
    else:
        fourcc = int(cap.get(cv2.cv.CV_CAP_PROP_FOURCC))  # Windows

    # 输出视频（记录跟踪过程）
    '''
    out_video_clone = out_video.replace('.','_alert' + '_' + datetime.now().strftime("%d%m%y_%H%M%S") + '.')
    out_clone_timer = 0
    out_clone_pending = False
    if out_video != "None":
        out = cv2.VideoWriter(out_video, fourcc, fps, (w, h))
        out_clone = cv2.VideoWriter(out_video_clone, fourcc, fps, (w, h))
    '''
    # # 为之后的标记创建一个随机颜色矢量
    # color = np.random.randint(0, 255, (params['LKT']['max_points'], 3))
    #
    # # 加载VJ跟踪的数据（XML数据）
    # face_cascade = cv2.CascadeClassifier(params['VJ']['VJ_Classifier'])

    # 二 预处理过程

    # 取第一帧，选取ROI（感兴趣区域）并计算角点:
    frmIndex = 1
    ret, old_frame_pre = cap.read()
    if not ret:
        print("Error loading first frame")
        exit(1)

    if scale != 1:
        old_frame = cv2.resize(old_frame_pre, (0, 0), fx=scale, fy=scale)
    else:
        old_frame = old_frame_pre

    ROI = [0, 0, w, h]

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **params['LKT']['feature_params'])


    cars_prev = vehicleDetector.detection(old_frame)

    # cars_prev = face_cascade.detectMultiScale(old_frame, **params['VJ']['cascade_params'])
    # # cars_prev = cv2.groupRectangles(np.array(cars_prev_pre).tolist(), 3, 0.2)

    # mask = np.zeros_like(old_frame)

    # 三 主要处理过程
    print("")
    carsAvg = 0
    velAvg = 0
    lastAlertFrm = -1000
    value = {}
    while 1:

        # 读取下一帧
        print('\rProcessing frame ' + str(frmIndex)),
        ret, frame_pre = cap.read()
        if not ret:
            print("\nError loading frame %d" % frmIndex)
            break
        frmIndex += 1

        if scale != 1:
            frame = cv2.resize(frame_pre, (0, 0), fx=scale, fy=scale)
        else:
            frame = frame_pre

        cars = vehicleDetector.detection(frame)
        for x1, y1, x2, y2 in cars:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (55 / 255.0, 255 / 255.0, 155 / 255.0), 2)

        good_new_VJ, good_old_VJ = Match_Features(cars, cars_prev, params['VJ']['velocity_thr'],
                                                  params['VJ']['area_thr'])

        ROIs = []
        cars_num = 0
        VJ_velAvg = 0
        VJ_velCnt = 0
        for i, (new, old) in enumerate(zip(good_new_VJ, good_old_VJ)):

            a, b, w1, h1 = new.ravel()
            c, d, w0, h0 = old.ravel()

            vx = abs((a + w1 / 2) - (c + w0 / 2))
            vy = abs((b + h1 / 2) - (d + h0 / 2))

            velocity = np.sqrt(vx ** 2 + vy ** 2)

            # 检测目标移动且处于ROI中
            if velocity > params['VJ']['velocity_thr'] and IsInROI(a, b, ROI):
                ROIs.append(new)


                # if debug_verbose_en:
                #     cv2.rectangle(frame, (a, b), (a + w1, b + h1), color=(255,0,0), thickness=1)
                #     cv2.putText(frame, "%d" % i, (a + w1/2, b-5),
                #                 cv2.FONT_HERSHEY_PLAIN, 1.0, (255,0,0), lineType=line_type)

                cars_num += 1
                VJ_velAvg = (VJ_velAvg * VJ_velCnt + velocity) / (VJ_velCnt + 1)
                VJ_velCnt += 1

            # if debug_verbose_en:
            #     print ('VJ: Frame %d - feature #%d: (x,y,w,h)=(%d,%d,%d,%d) velocity=%.2f' % (frmIndex, i, a, b, w1, h1, velocity))

        # 交通事故发生的检测
        for i1, roi1 in enumerate(ROIs):
            for i2, roi2 in enumerate(ROIs):

                if i1 > i2:

                    d_ = np.sqrt((roi1[0] - roi2[0]) ** 2 + (roi1[1] - roi2[1]) ** 2)
                    w_ = min(roi1[2], roi2[2])
                    h_ = min(roi1[3], roi2[3])

                    if (w_ / 2 < d_ < w_) and not nestedROIs(roi1, roi2):
                        a = min(roi1[0], roi2[0])
                        b = min(roi1[1], roi2[1])
                        if lastAlertFrm <= 0:
                            lastAlertFrm = frmIndex

                        if (frmIndex - lastAlertFrm) > inter_frames_alerts:
                            value = {'alType': 'Accident',
                                     'alTime': frmIndex,
                                     'alX': a + w_ / 2,
                                     'alY': b + h_ / 2}

                            res['alerts'][str(len(res['alerts']))] = value
                            lastAlertFrm = -1
                            '''
                            if debug_verbose_en:
                                print 'Alert detected: (x,y)=(%d,%d)' % (a+w/2,b+h/2)
                            '''

                    '''
                    if debug_verbose_en:
                        print 'ROI #%d --> (%d,%d,%d,%d)' % (i1,roi1[0],roi1[1],roi1[2],roi1[3])
                    '''
        # 绘制过程
        '''
        for key, value in res['alerts'].iteritems():
            a = value['alX']
            b = value['alY']
            w_ = 70
            h_ = 70
            cv2.rectangle(frame, (a, b), (a + w_, b + h_), color=(0,0,255), thickness=4)
            cv2.putText(frame, "ALERT #%s" % key, (a, b-5),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), lineType=line_type)

        img = cv2.add(frame, mask)
        if platform == 'darwin':
            draw_str(img, (20, 20), 'Frame #%d , LKT=%d , VJ=%d , Cars=%d , Alerts=%d' %
                     (frmIndex, len(good_new_LKT), cars_num, len(ROIs), len(res['alerts'])))
            draw_str(img, (20, 40), 'Press \'q\' to quit')
            cv2.imshow('frame', img)
        '''
        '''
        # Save frame:
        if out_video != "None":
            out.write(img)
            out_clone.write(img)
        '''

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if src > 0 and frmIndex >= framesNum:
            break

        # 更新之前的检测车辆
        cars_prev = cars

        # 更新统计量
        # carsAvg_LKT = (carsAvg * (frmIndex - 1) + len(good_new_LKT)) / frmIndex
        carsAvg_VJ = (carsAvg * (frmIndex - 1) + cars_num) / frmIndex
        # carsAvg = (carsAvg_LKT + carsAvg_VJ) / 2

        # velAvg_LKT = (velAvg * (frmIndex - 1) + LKT_velAvg) / frmIndex
        velAvg_VJ = (velAvg * (frmIndex - 1) + VJ_velAvg) / frmIndex
        # velAvg = (velAvg_LKT + velAvg_VJ) / 2

    # Done:


    cap.release()
    '''
    if out_video != "None":
        print ('Results --> %s' % out_video)
        out.release()
        out_clone.release()
    '''
    '''
    if platform == 'linux2':
        call(['ffmpeg', 'i '+ out_video,'filter:v "setpts=12.0*PTS" ' + out_video.replace("mov","mp4")])
    '''
    cv2.destroyAllWindows()

    res['carsAvg'] = str(carsAvg)
    res['velAvg'] = str(format(velAvg, '.2f'))

    return res

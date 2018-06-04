import os
os.environ['GLOG_minloglevel'] = '2'
import sys
reload(sys)
sys.path.append('/home/aythior/install_env/caffe-master/python')
sys.setdefaultencoding('utf8')

import functools
import caffe 
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

def singleton(cls):
    instances = {}

    @functools.wraps(cls)
    def getinstance(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return getinstance


@singleton
class VehicleDetector(object):
    def __init__(self, caffe_root, caffe_model, deploy_file, labels_file):
        caffe.set_mode_cpu()
        #caffe.set_device(0)
        caffe.set_mode_gpu()
        self.caffe_root = caffe_root
        self.caffe_model = caffe_model
        self.labels_file =labels_file
        self.deploy = deploy_file
        mu = np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        mu = mu.mean(1).mean(1)
        self.net = caffe.Net(self.deploy, 
                self.caffe_model,
                caffe.TEST) 
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))
        self.transformer.set_mean('data', mu)
        # self.transformer.set_raw_scale('data', 255)
        # self.transformer.set_channel_swap('data', (2,1,0))
        with open(labels_file, 'r') as file:
            self.labelmap = caffe_pb2.LabelMap()
            text_format.Merge(str(file.read()), self.labelmap)


    def detection(self, im):

        # im = caffe.io.load_image(img)
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', im)

        start = time.clock()
        self.net.forward()
        end = time.clock()
        print('detection time: %f s' % (end - start))

        loc = self.net.blobs['detection_out'].data[0][0]
        confidence_threshold = 0.3
        # cv2.putText(im, img.split('/')[-1], (55, 55), cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)

        result = []
        for l in range(len(loc)):
            confidence = loc[l][2]
            if confidence >= confidence_threshold:
                xmin = int(loc[l][3] * im.shape[1])
                ymin = int(loc[l][4] * im.shape[0])
                xmax = int(loc[l][5] * im.shape[1])
                ymax = int(loc[l][6] * im.shape[0])
                # class_name = self.labelmap.item[int(loc[l][1])].display_name
                # result.append((class_name, confidence, xmin, ymin, xmax, ymax))
                result.append((xmin, ymin, xmax, ymax))
        
        return np.array(result)


# caffe.set_mode_cpu()
# #caffe.set_device(0)
# caffe.set_mode_gpu()

# caffe_root = './'
# caffemodel = caffe_root + 'models/VGGNet/vehicle/SSD_300x300/VGG_vehicle_SSD_300x300_iter_31971.caffemodel'
# deploy = caffe_root + 'models/VGGNet/vehicle/SSD_300x300/deploy.prototxt'


# #img_root = caffe_root + 'data/VOCdevkit/VOC2007/JPEGImages/'
# labels_file = caffe_root + 'data/vehicle/labelmap_vehicle.prototxt'

# net = caffe.Net(deploy, 
#                 caffemodel,
#                 caffe.TEST) 

# mu = np.load('/home/aythior/install_env/caffe-master/python/caffe/imagenet/ilsvrc_2012_mean.npy')
# mu = mu.mean(1).mean(1)

# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_transpose('data', (2,0,1))
# transformer.set_mean('data', mu)
# transformer.set_raw_scale('data', 255)
# transformer.set_channel_swap('data', (2,1,0))
# #class_img/IMG_20171123_135007.jpg  
# #  inflating: class_img/IMG_20171123_135357.jpg  
# #  inflating: class_img/IMG_20171123_135537.jpg  
# #  inflating: class_img/IMG_20171123_135626.jpg  
# #  inflating: class_img/IMG_20171123_135641.jpg  
# #  inflating: class_img/IMG_20171123_135722.jpg
# #while 1:
# #    img_num = raw_input("Enter Img Number: ")
# #    if img_num == '': break
# #    img = img_root + '{:0>6}'.format(img_num) + '.jpg'
# #if __name__ == '__main__':
# #img = 'examples/images/vehicle_test/object_2017_0091425.jpg'
# path = "examples/images/vehicle_test/"
# files= os.listdir(path)
# for file in files:
#     img = path + file
#     detection(img,net,transformer,labels_file)

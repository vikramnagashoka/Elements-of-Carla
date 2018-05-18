from styx_msgs.msg import TrafficLight

import numpy as np
import cv2
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from PIL import Image
import os
import six.moves.urllib as urllib
from collections import defaultdict
from io import StringIO
import time
from glob import glob

cwd = os.path.dirname(os.path.realpath(__file__))

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        #self.threshold = threshold
        #self.hw_ratio = hw_ratio    #height_width ratio
        #print('Initializing classifier with threshold =', self.threshold)
        self.signal_classes = ['Red', 'Green', 'Yellow']
       # self.signal_status = TrafficLight.UNKNOWN
        self.signal_status = None

        self.traffic_box = None

        self.num_pixels = 25
        # Define red pixels in hsv color space
        # self.lower_red_1 = np.array([0,  70, 50],   dtype = "uint8")
        # self.upper_red_1 = np.array([10, 255, 255], dtype = "uint8")

        # self.lower_red_2 = np.array([170,  70,  50], dtype = "uint8")
        # self.upper_red_2 = np.array([180, 255, 255], dtype = "uint8")

        os.chdir(cwd)

        self.class_model = load_model('tl_classifier_simulator.h5') #switched to model 5 for harsh light
        self.graph = tf.get_default_graph()

        #tensorflow localization/detection model
        model = 'ssd_mobilenet_v1_coco_11_06_2017' #was 'ssd_inception_v2_coco_11_06_2017'
        PATH_TO_CKPT = model + '/frozen_inference_graph.pb'
        # setup tensorflow graph
        self.detection_graph = tf.Graph()
        # configuration for possible GPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
               serialized_graph = fid.read()
               od_graph_def.ParseFromString(serialized_graph)
               tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
              # Each box represents a part of the image where a particular object was detected.
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
              # Each score represent how level of confidence for each of the objects.
              # Score is shown on the result image, together with the class label.
            self.scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')


    def box_normal_to_pixel(self, box, dim):

        height, width = dim[0], dim[1]
        box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
        return np.array(box_pixel)

    def get_bounding_box(self, image):
        with self.detection_graph.as_default():
              image_expanded = np.expand_dims(image, axis=0)
              (boxes, scores, classes, num_detections) = self.sess.run(
                  [self.boxes, self.scores, self.classes, self.num_detections],
                  feed_dict={self.image_tensor: image_expanded})

        boxes=np.squeeze(boxes)  # bounding boxes
        classes = list(np.squeeze(classes))  # classes
        scores = np.squeeze(scores) # confidence

        idx = next((i for i, v in enumerate(cls) if v == 10.), -1)
        confidence = scores[idx]
        cls_idx = classes[idx] 

        if idx == -1:
            box=[0, 0, 0, 0]
        elif scores[idx]<=0.3: #updated site treshold to 0.01 for harsh light
            box=[0, 0, 0, 0]
        else:
            dim = image.shape[0:2]

            #convert to pixels
            height, width = dim[0], dim[1]
            box = [int(boxes[0]*height), int(boxes[1]*width), int(boxes[2]*height), int(boxes[3]*width)]
            
            #box = self.box_normal_to_pixel(boxes[idx], dim)
            box_h = box[2] - box[0]
            box_w = box[3] - box[1]
            #ratio = box_h/(box_w + 0.01)
                  # if the box is too small, 20 pixels for simulator
            if (box_h <20) or (box_w<20):
                box =[0, 0, 0, 0]
                print('box too small!', box_h, box_w)
              # if the h-w ratio is not right, 1.5 for simulator, 0.5 for site
            # elif (ratio < self.hw_ratio):
            #     box =[0, 0, 0, 0]
            #     print('wrong h-w ratio', ratio)
            # else:
            #     print(box)
            #     print('localization confidence: ', scores[idx])
                 #****************end of corner cases***********************
            self.traffic_box = box

        return box

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        img_resize=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = np.expand_dims(img_resize, axis=0).astype('float32')
        # Normalization
        img_resize/=255.
        # Prediction
        with self.graph.as_default():
            signal_color_prob = self.class_model.predict(img_resize)
            
            color = self.signal_classes[np.argmax(signal_color_prob)]
            
            self.signal_status = color
        # uncomment the following in real test
        if color == 'Red':
            self.signal_status = TrafficLight.RED
        elif color == 'Green':
            self.signal_status = TrafficLight.GREEN
        elif color == 'Yellow':
            self.signal_status = TrafficLight.YELLOW

        return self.signal_status


        

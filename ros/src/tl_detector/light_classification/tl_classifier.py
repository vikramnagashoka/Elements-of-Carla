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

        
        self.signal_classes = ['Red', 'Yellow', 'Green', 'None']
        self.signal_status = TrafficLight.UNKNOWN
        #self.signal_status = None

        self.traffic_box = None

        self.num_pixels = 25
        

        os.chdir(cwd)

        self.class_model = load_model('tl_classifier_simulator_one.h5') 
        self.graph = tf.get_default_graph()

        #tensorflow localization/detection model
        model = 'ssd_mobilenet_v1_coco_11_06_2017' 
        PATH_TO_CKPT = model + '/frozen_inference_graph.pb'
        
        # setup tensorflow graph
        self.detection_graph = tf.Graph()
        
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

              
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')

    def get_bounding_box(self, image):
        with self.detection_graph.as_default():
              image_expanded = np.expand_dims(image, axis=0)
              (boxes, scores, classes, num_detections) = self.sess.run(
                  [self.boxes, self.scores, self.classes, self.num_detections],
                  feed_dict={self.image_tensor: image_expanded})

        boxes=np.squeeze(boxes)                 #bounding boxes
        classes = list(np.squeeze(classes))     
        scores = np.squeeze(scores) 

        idx = next((i for i, v in enumerate(classes) if v == 10.), -1)
        confidence = scores[idx]
        cls_idx = classes[idx] 

        if idx == -1:
            box=[0, 0, 0, 0]
        elif scores[idx]<=0.3:
            box=[0, 0, 0, 0]
        else:
            dim = image.shape[0:2]

            
            height, width = dim[0], dim[1]
            
            #convert box co ordinates to pixels

            box = [int(boxes[0][0]*height), int(boxes[1][0]*width), int(boxes[0][2]*height), int(boxes[0][3]*width)]
            
            
            box_h = box[2] - box[0]
            box_w = box[3] - box[1]
                  
            #If the first bounding box is too small, iterate over remaining set of boxes to find an optimal one.
            box_index = 1
            while (((box_h <20) or (box_w<20)) and box_index < boxes.shape[0]):
                box =[0, 0, 0, 0]
                
                box = [int(boxes[box_index][0]*height), int(boxes[box_index][0]*width), int(boxes[box_index][2]*height), int(boxes[box_index][3]*width)]
                box_index += 1

            box_h = box[2] - box[0]
            box_w = box[3] - box[1]

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
            print('color is: ', color)
            
            self.signal_status = color
        # uncomment the following in real test
        if color == 'Red':
            self.signal_status = TrafficLight.RED
        elif color == 'Green':
            self.signal_status = TrafficLight.GREEN
        elif color == 'Yellow':
            self.signal_status = TrafficLight.YELLOW
        else:
            self.signal_status = TrafficLight.UNKNOWN

        return self.signal_status


        

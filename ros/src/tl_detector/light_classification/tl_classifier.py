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
        #load classifier
        
        self.signal_status = TrafficLight.UNKNOWN

        os.chdir(cwd)

        # tensorflow localization/detection model
        PATH_TO_CKPT = 'sim_model.pb'
        
        # setup tensorflow graph
        self.detection_graph = tf.Graph()
        
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
               serialized_graph = fid.read()
               od_graph_def.ParseFromString(serialized_graph)
               tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')


    def get_class(self, image):
        with self.detection_graph.as_default():
              image_expanded = np.expand_dims(image, axis=0)
              (boxes, scores, classes, num_detections) = self.sess.run(
                  [self.boxes, self.scores, self.classes, self.num_detections],
                  feed_dict={self.image_tensor: image_expanded})

        boxes = np.squeeze(boxes)                 
        classes = list(np.squeeze(classes))     
        scores = np.squeeze(scores)

        if len(scores) == 0 or scores[0] < 0.2:
            self.signal_status = TrafficLight.UNKNOWN
            return self.signal_status
        else:
            result_class = classes[0] 
            if result_class == 2:
                self.signal_status = TrafficLight.RED
            elif result_class == 1:
                self.signal_status = TrafficLight.GREEN
            elif result_class == 3:
                self.signal_status = TrafficLight.YELLOW

            return self.signal_status



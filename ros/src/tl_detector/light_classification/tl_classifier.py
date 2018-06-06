import os
import numpy as np
import tensorflow as tf
from styx_msgs.msg import TrafficLight


CWD = os.path.dirname(os.path.realpath(__file__))


class TLClassifier(object):
    def __init__(self):
        #load classifier

        os.chdir(CWD)

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

            self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')


    def get_class(self, image):
        with self.detection_graph.as_default():
            image_expanded = np.expand_dims(image, axis=0)
            (scores, classes) = self.sess.run(
                [self.scores, self.classes],
                feed_dict={self.image_tensor: image_expanded}
            )

        classes = list(np.squeeze(classes))
        scores = np.squeeze(scores)

        if len(scores) == 0 or scores[0] < 0.2:
            return TrafficLight.UNKNOWN
        else:
            result_class = classes[0]
            if result_class == 2:
                return TrafficLight.RED
            elif result_class == 1:
                return TrafficLight.GREEN
            elif result_class == 3:
                return TrafficLight.YELLOW

            return TrafficLight.UNKNOWN

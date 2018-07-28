# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Modified by Dmitry Pechyony, 2018

"""Convert the dataset of real camera images to TFRecord file. Each image has a single 
   label with the color of the traffic light. If there is no visible traffic light then 
   the image has no label.

Example usage:
    python create_real_camera_tf_record_image_label.py --data_dir=<directory of images> \
        --output_path=<location and name of the output file> \
        --label_map_path=<path to label map file> 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import glob

import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

# definitions of labels

# bag1
labels = np.empty(shape=985,dtype=object)
labels[0:18] = 'red'
labels[18:140] = 'green'
labels[140:201] = 'yellow'
labels[201:323] = 'red'
labels[323:445] = 'green' 
labels[445:454] = 'yellow'
labels[454:985] = 'nothing'

# bag2
#labels = np.empty(shape=1193,dtype=object)
#labels[0:77] = 'nothing'
#labels[77:192] = 'green'
#labels[192:253] = 'yellow'
#labels[253:374] = 'red'
#labels[374:496] = 'green'
#labels[496:557] = 'yellow'
#labels[557:657] = 'red'
#labels[657:1193] = 'nothing'


def dict_to_tf_example(full_path, label_text, label=-1):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    full_path: full path to JPEG file 
    label_text: textual label of the image
    label: numeric label of the image
    
  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()
  filename = full_path.split('/')[-1]

  width = image.size[0]
  height = image.size[1]

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  difficult_obj = []
  truncated = []
  poses = []
  
  if label_text != 'nothing':
     label_text_list = dataset_util.bytes_list_feature([label_text.encode('utf8')])
     label_list = dataset_util.int64_list_feature([label])
  else:
     label_text_list = dataset_util.bytes_list_feature([])
     label_list = dataset_util.int64_list_feature([])

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': label_text_list,
      'image/object/class/label': label_list,
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))

  return example


def main(_):

  data_dir = FLAGS.data_dir
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  for full_path in glob.glob(data_dir + '/*.jpg'):

    idx = int(full_path.split('/')[-1].split('.')[0][4:]) 
    if idx % 10 == 5:
        logging.info('On image %d', idx)

        label_text = labels[idx]
        if (label_text != 'nothing'):
            label = label_map_dict[label_text]
            tf_example = dict_to_tf_example(full_path, label_text, label)
        else:
            tf_example = dict_to_tf_example(full_path, label_text)
        
        writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()

## Traffic light detection

### Environment Setup

We developed traffic light detection model using Object Detection API of Tensorflow. This model is deployed in [self-driving car](https://medium.com/udacity/how-the-udacity-self-driving-car-works-575365270a40) of Udacity, that has version 1.3 of Tensorflow and Titan X GPU. Hence, to prevent any potential incompatibility issues, we developed the model using the same version of Tensorflow. 

We used GPU-enabled version of Tensorflow to speed model training and to generate a GPU-aware model. To train the model, we used [NC6 virtual machine](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu) in Azure cloud. This machine has K80 GPU of NVIDIA. Even with such powerful GPU, training the model took several hours.

Object Detection API doesn't have the same versioning system as Tensorflow. Therefore, in order to prevent conflicts between Tensorflow and Object Detection API,  we had to determine the last commit to Object Detection API that is compatible with version 1.3 of Tensorflow. According to [Releases](https://github.com/tensorflow/tensorflow/releases?after=v1.4.0-rc1) section of Tensorflow repository, Version 1.3 of Tensorflow was released at 8/16/2017, the next version, 1.3.1, was released at 9/26/2017. According to [this](https://github.com/tensorflow/models/commits/master?after=86ac7a47f03c08112a93b1a18be8e23b8989c4e9+174&path%5B%5D=research&path%5B%5D=object_detection) page, the last commit to Object Detection API before 9/26/2017 was 4a705e08ca8f28bd05bdc2287211ad8aba0bf98a and it was done at 9/22/2017. Hence we cloned the repository of Objecty Detection API and checked out this commit:

    git clone https://github.com/tensorflow/models
    cd models
    git checkout 4a705e08ca8f28bd05bdc2287211ad8aba0bf98a   

As part of installing Object Detection API, we added the following line to `~/.bashrc` file:

    export PYTHONPATH=$PYTHONPATH:<local direcotry of models repo>/models/research:<local directory of models repo>/models/research/slim

### Datasets

We used 3 datasets:   
* Dataset 1: the [dataset](https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0) created by [Alex Lechner](https://github.com/alex-lechner). This dataset has 917 images from simulator and 155 images from the test track. 
* Dataset 2: the [dataset](https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view?usp=sharing) created by [Vatsal Srivastava](https://github.com/coldKnight). This dataset has 277 images from simulator and 159 images from the test track.
* Dataset 3: our own dataset of images extracted from [ROS bag file](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing) provided by Udacity. This dataset has 297 images from the test track and is available [here](https://www.dropbox.com/s/ii4ddadp7lih7b4/dataset_ros.zip?dl=0).

Object Detection API expects the dataset to be in a single-binary-file TFRecord format. Fortunately, the first two datasets have already TFRecord files, one for simulated and one for real images. In the rest of this section we show how we created the last dataset.

ROS bag file can be replayed using `rosbag` tool of ROS.  `image_view` utility of ROS allows to save images published in a given topic. We created a launch script, called `export.launch` that runs `rosbag` and `image_view`:

    <launch>
        <node pkg="rosbag" type="play" name="rosbag" args="<full path to ROS bag file>" />
        <node name="extract" pkg="image_view" type="image_saver" respawn="false" output="screen" cwd="ROS_HOME">
        <remap from="image" to="image_raw" />
        </node>
    </launch>

The parameter args in the second line of launch should have a full path to ROS bag file. `export.launch` script is stored in `ros\launch` directory. To extract the images from ROS bag file we just need to run

    roslaunch export.launch

The images are extracted to ~/.ros directory. Altogether, we extracted 2030 JPEG images.

We used [labelImg](https://github.com/tzutalin/labelImg) utility to annotate 297 images that have a visible traffic light. This program creates XML files with annotations. We saved all annotation in `labels` directory that is placed in the directory of images. Then we used our own `create_real_camera_tf_record.py` utility to convert annotated images to a single TFRecord file:

    python create_real_camera_tf_record.py --data_dir=<path to the directory of images> --annotations_dir=labels --output_path=<path to the output file> --label_map_path=../label_mapping_lowercase.pbtxt 

`create_real_camera_tf_record.py` is a simplified version of `create_pascal_tf_record.py` utility of Object Detection API and is stored in `ros/src/tl_detector/model_training` directory.  

`create_real_camera_tf_record.py` has a number of parameters. `data_dir` should point to the directory of images, `annotations_dir` should be a name of directory with XML files. Notice that due to limitations of `create_real_camera_tf_record.py`, `annotations_dir` should be a directory within `data_dir`. `output_path` is a path to the output TFRecord file. Finally, `label_map_path` is a path to the file that maps categorical labels in annotations files to numberic labels that will be used by Tensorflow. We used the following mapping:

    item {
        id: 1
        name: 'green'
    }
    item {
        id: 2
        name: 'red'
    }
    item {
        id: 3
        name: 'yellow'
    }

This mapping is consistent with the mappping in two other datasets mentioned above. The mapping data stored in `label_mapping_lowercase.pbtxt` file in `ros/src/tl_detector/model_training` directory. 

### Training

Traffic light detection module get a new image with the frequency of 15Hz. Hence, to process images in real-time, the traffic light detection model should score a new image within 1/15 seconds = 66 milliseconds. 
Our model is based on pre-trained SSD-MobileNet-V1-COCO model that is available [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz). In particular our network has the same architecture as SSD-MobileNet, except the final prediction layer. Also, the weights in all layers, except the final one, were initialized with the corresponding values in SSD-MobileNet-V1-COCO. We chose SSD-MobileNet-V1-COCO because it was the only model with real-time inference that is compatible with version 1.3.0 of Tensorflow. The zipped model has 3 checkpoint files (with ckpt extension). We placed these files in a separate `models` directory. We also copied configuration file of the model, `<path to Object Detection API>/models/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config`, to the current directory. 

The training process assumes that training and test sets, as well as the label mapping file are stored in `data` directory. We did a number of changes in model configuration file to specify the locations of dataset, checkpoint and label mapping files. The line 

    fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"

was replaced with 

     fine_tune_checkpoint: "models/model.ckpt"
Next, the line 

    input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record"
in the input_reader section of config file was replaced with

    input_path: "data/<dataset name>.record"
Finally, the line

    label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
in the input_reader section of config file was replaced with 

    label_map_path: "data/label_mapping_lowercase.pbtxt"
We didn't use evaluation functionality of Object Detection API and hence we didn't change eval_config and eval_input_reader sections of config file.

Since the original model was trained with 90 classes and in our dataset we have only 3 classes of objects (green, yellow and red), we changed the architecture of the final prediction layer by replacing the line  

     num_classes: 90   
in model configuration file with  

     num_classes: 3

Finally, we changed the maximal number of epochs from 200000 to 20000 by replacing 

    num_steps: 200000
with

    num_steps: 20000

We would like to save the model after each epoch and at the end of the training process choose the one with the lowest training loss value. Unfortunately, Object Detection API can save a model only in fixed time intervals. We saved the model every 0.3 seconds. This resulted in saving the model after approximately every second epoch. We tried to reduce the time saving interval from 0.3 seconds to smaller values, but this also resulted in saving the model after approximately every second epoch. 

At the end of the training process we had about 10000 models. At this point  we chose manually the model with the smallest loss.  

To save the model every 0.3 seconds and to keep the last saved 20000 models, we modified `models/research/object_detection/trainer.py` in two places:  
* we added a parameter max_to_keep=20000 when initializing tf.train.Saver (line 281 in `trainer.py`)  
* we added parameters save_summaries_secs=0.3 and save_interval_secs=0.3 when calling slim.learning.train (line 284 in `trainer.py`).

The training process is managed by `train.py` file provided by Object Detection API in `models/research/object_detection` folder. We launched the training process using the following command:

    nohup python <path to Object Detection API>/models/research/object_detection/train.py --train_dir=./models/train --pipeline_config_path=<model config file> >& out.txt &

The training takes around 7 hours. `nohup` and the trailing `&` are needed for running the training in the background and uninterrupted running when we log out of virtual machine.  

When we ran the training for the first time, it failed with the error

    Traceback (most recent call last):
    File "train.py", line 200, in <module>
      tf.app.run()
    File  "/anaconda/envs/py35/lib/python3.5/site-packages/tensorflow/python/platform/app.py", line 48, in run
    _sys.exit(main(_sys.argv[:1] + flags_passthrough))
    File "train.py", line 196, in main
       worker_job_name, is_chief, FLAGS.train_dir)
    File "/datadrive/models/research/object_detection/trainer.py", line 186, in train
      data_augmentation_options)
    File "/datadrive/models/research/object_detection/trainer.py", line 59, in _create_input_queue
      tensor_dict = create_tensor_dict_fn()
    File "/datadrive/models/research/object_detection/builders/input_reader_builder.py", line 61, in build
      min_after_dequeue=input_reader_config.min_after_dequeue)
    File "/anaconda/envs/py35/lib/python3.5/site-packages/tensorflow/contrib/slim/python/slim/data/parallel_reader.py", line 214, in parallel_read
    name='filenames')
    ...
    TypeError: Expected string, got ['data/images_ros_all.record'] of type 'RepeatedScalarContainer' instead.

This is a [known issue](https://github.com/tensorflow/models/issues/2757) of Object Detection API that was fixed in later commits. In our case we fixed it by changing line 53 in
`models/research/object_detection/builders/input_reader_builder.py` file from

     config.input_path,

to

     config.input_path[:],

#### Model for simulator images
We used simulator images of Dataset 1 as a training and evaluation set. Simulator images from Dataset 2 were a test set. The configuration file for training this model is stored in repo in `ros/src/tl_detector/model_training/ssd_mobilenet_v1_coco.config_sim` file. The model with the smallest loss was obtained after epoch 15339 and had a loss 0.6341 . We also stored the training log file in `ros/src/tl_detector/model_training/training_logs/log_sim.txt`.

#### Model for real images
We concatenated real images from all 3 datasets into a single training set. This training set can be downloaded from [here](https://www.dropbox.com/s/s28zz6ia9kafesw/images_ros_all.record?dl=0).  The configuration file for training this model is stored in repo in `ssd_mobilenet_v1_coco.config_sim` file. The model with the smallest loss was obtained after epoch 6651 and had a loss 0.4667.

### Deployment

After training a model we exported it using the command

    python <path to Object Detection API>/models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path ./ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix ./models/train/model.ckpt-<best epoch number> --output_directory ./fine_tuned_model

This command creates a `fine_tuned_model/frozen_inference_graph.pb` file. We renamed this file to either `sim_model.pb` or `real_model.pb`, depending on the type of the model, and copied it to `ros/src/tl_detector/light_classification` folder.

### Evaluation

In the evaluation process we considered only the detected box with the highest score. If this score is at least 0.1 then we defined that the model detected a light in the image, otherwise we define that the model does not detect any light. The same logic is used by traffic light detector module when applying the model in real-time. 

#### Model for simulator images 

We evaluated the model over random 5 images from the training set and random 10 images from the test set. These images are stored in `ros/src/tl_detector/model_training/test_images` folder. The evaluation  is done in `ros/src/tl_detector/model_training/object_detection_evaluation_sim.ipynb` notebook. The model has perfect detections in 14 out of 15 images. In a single problematic image all lights are very small. But nevertheless in that image the detection with the highest score is correct (red light) and this is sufficient for practical purposes.  

#### Model for real images

We evaluated the model over all 2030 images extracted from  [ROS bag file](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing) that was provided by Udacity. 297 of these images are also in the training set. The evaluation is done in `ros/src/tl_detector/model_training/object_detection_evaluation_real.ipynb` notebook. 

For the purpose of evaluation, we labelled manually all images with the labels 'green', 'yellow', 'red', 'no light'. Unlike training labels, these are categorical labels and not bounding boxes. During evaluation we computed detection error and classification error. Detection error is the number of images that satisfy either of these two conditions:

*  there was no visible light, but the model detected a bounding box with a yellow or red light light.  
* there is a visible light, but the model did not detect it

Notice that when there is no visible light but the model detected a green light, the car continues to drive as usual and we do not count this case as a wring detection. Classification error is the number of images where there is a visible light, the model detected it, but classified with the wrong color. 

Our model has 11 images with detection error and 8 images with classification error. Overall the model generated correct detections in 2011/2030*100%=99% of the images.
## Traffic light detection

### Environment Setup

We developed traffic light detection model using Object Detection API of Tensorflow. This model is deployed in [self-driving car](https://medium.com/udacity/how-the-udacity-self-driving-car-works-575365270a40) of Udacity, that has version 1.3 of Tensorflow and Titan X GPU. Hence, to prevent any potential incompatibility issues, we developed the model using the same version of Tensorflow. 

We used GPU-enabled version of Tensorflow to speed model training and to generate a GPU-aware model. To train the model, we used [NC6 virtual machine](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu) in Azure cloud. This machine has K80 GPU of NVIDIA. Even with such powerful GPU, training the model took several hours.

Object Detection API doesn't have the same versioning system as Tensorflow. Therefore, in order to prevent conflicts between Tensorflow and Object Detection API,  we had to determine the last commit to Object Detection API that is compatible with version 1.3 of Tensorflow. According to [Releases](https://github.com/tensorflow/tensorflow/releases?after=v1.4.0-rc1) section of Tensorflow repository, Version 1.3 of Tensorflow was released at 8/16/2017, the next version, 1.3.1, was released at 9/26/2017. According to [this](https://github.com/tensorflow/models/commits/master?after=86ac7a47f03c08112a93b1a18be8e23b8989c4e9+174&path%5B%5D=research&path%5B%5D=object_detection) page, the last commit to Object Detection API before 9/26/2017 was 4a705e08ca8f28bd05bdc2287211ad8aba0bf98a and it was done at 9/22/2017. Hence we cloned the repository of Objecty Detection API and checked out this commit:

    git clone https://github.com/tensorflow/models
    cd models
    git checkout 4a705e08ca8f28bd05bdc2287211ad8aba0bf98a   

Finally, as part of installing Object Detection API, we added the following line to `~/.bashrc` file:

    export PYTHONPATH=$PYTHONPATH:<local direcotry of models repo>/models/research:<local directory of models repo>/models/research/slim

### Datasets

We used 3 datasets:   
* the [dataset](https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0) created by [Alex Lechner](https://github.com/alex-lechner). This dataset has 917 images from simulator at 155 images from the test track. 
* the [dataset](https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view?usp=sharing) created by [Vatsal Srivastava](https://github.com/coldKnight). This dataset has 277 images from simulator and 159 images from the test track.
* our own dataset of images extracted from [ROS bag file](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing) provided by Udacity. This dataset has 297 images from the test track and is available here (link TBD).

Object Detection API expects the dataset to be in a single binary file TFRecord format. Fortunately, the first two datasets have already TFRecord files, one for simulated and one for real images. In the rest of this section we show how we created the last dataset.

ROS file can be replayed using `rosbag` tool of ROS.  `image_view` utility of ROS allows to save images published in a given topic. We created a launch script, called `export.launch` that runs `rosbag` and `image_view`:

    <launch>
        <node pkg="rosbag" type="play" name="rosbag" args="<full path to ROS bag file>" />
        <node name="extract" pkg="image_view" type="image_saver" respawn="false" output="screen" cwd="ROS_HOME">
        <remap from="image" to="image_raw" />
        </node>
    </launch>

The parameter args in the second line of launch should have a full path to ROS bag file. `export.launch` script is stored in `ros\launch` directory. To extract the images we just need to run

    roslaunch export.launch

The images are extracted to ~/.ros directory. Altogether, we extracted 2030 JPEG images.

We used [labelImg](https://github.com/tzutalin/labelImg) utility to annotate 297 images that have a visible traffic light. This program creates XML files with annotations. We saved all annotation in `labels` directory that is placed in the directory of images. Then we used our own `create_real_camera_tf_record.py` utility to convert annotated images to a single TFRecord file:

    python create_real_camera_tf_record.py --data_dir=<path to the directory of images> --annotations_dir=labels --output_path=<path to the output file> --label_map_path=../label_mapping_lowercase.pbtxt 

`create_real_camera_tf_record.py` is a simplified version of `create_pascal_tf_record.py` utility of Object Detection API and is stored in `ros/src/tl_detector/model_training` directory.
`create_real_camera_tf_record.py` has a number of parameters. `data_dir` should point to the directory of images, `annotations_dir` should be a name of directory with XML files. Notice that due to limitations of `create_real_camera_tf_record.py`, `annotations_dir` should be a directory within `data_dir`. `output_path` is a path to the output TFRecord file. Finally, label_map_path is a path to the file that maps categorical labels in annotations files to numberic labels that will be used by Tensorflow. We used the following mapping:

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

#### Model for simulated images

#### Model for real images

### Evaluation

### Model deployment

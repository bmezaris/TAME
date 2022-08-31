# TAME

This repository hosts the code and data lists for our learning-based eXplainable AI (XAI) method called TAME, for Convolutional Neural Networks (CNN) image classifiers. Our methods receive as input an image and a class label and produce as output the image regions that the CNN has focused on in order to infer this class. TAME uses an attention mechanism (AM), trained  end-to-end  along  with the original (frozen) CNN, to derive class activation maps from feature map sets extracted from selected layers . During training, the generated attention maps of the AM are applied to the inputs. The AM weights are updated by applying backpropagation on a multi-objective loss function to optimize the appearance of the attention maps (minimize high-frequency variation and attention mask area) and minimize the cross-entropy loss. This process forces the AM to learn the image regions responsible for the CNN’s output. Two widely-used evaluation metrics, Increase in Confidence (IC) and Average Drop (AD), are used for evaluation.
- This repository contains the code for training, evaluating and applying TAME, using VGG-16 or ResNet-50 as the pre-trained backbone network along with the Attention Mechanism and our selected loss function. There is also a guide on applying TAME to any CNN classifier.
- Instead of training, the user can also use a pretrained attention mechanism for the pretrained VGG-16 or ResNet-50 classifiers in [TAME/snapshot/](https://github.com/bmezaris/TAME/tree/main/snapshots/).
- In [TAME/datalist/ILSVRC](https://github.com/bmezaris/TAME/tree/main/datalist/ILSVRC) there are text files with annotations for training VGG-16 and ResNet-50 (VGG-16_train.txt, ResNet50_train.txt) and text files with annotations for 2000 randomly selected images to be used at the validation stage (Evaluation_2000.txt) and 2000 randomly selected images (exclusive of the previous 2000) for the evaluation stage (Test_2000.txt) the L-CAM methods.
- The ILSVRC 2012 dataset images should be downloaded by the user manually.
- The required packages for this project are contained in requirements.txt. This project was developed with Python 3.8

## Data preparation
Download [here](https://image-net.org/) the training and evaluating images for the ILSVRC 2012 dataset, then extract folders and sub-folders and place the extracted folders (ILSVRC2012_img_train, ILSVRC2012_img_val) in the dataset/ILSVRC2012_img_train and dataset/ILSVRC2012_img_val folders. The folder structure of the image files should look like this:
```
dataset
    └── ILSVRC2012_img_train
        └── n01440764
            ├── n01440764_10026.JPEG
            ├── n01440764_10027.JPEG
            └── ...
        └── n01443537
        └── ...
    └── ILSVRC2012_img_val
        ├── ILSVRC2012_val_00000001.JPEG
        ├── ILSVRC2012_val_00000002.JPEG
        └── ...
```
If you have already downloaded the ILSVRC 2012 dataset and don't want to change its location, edit the paths in the file scripts/bash scripts/pc_info.sh to the location of your dataset.

## Initial Setup
Check that you have a working Python 3 and pip installation before proceeding.
- Clone this repository:
~~~
git clone https://github.com/bmezaris/TAME
~~~
- Go to the locally saved repository path:
~~~
cd TAME
~~~
- Create and activate a virtual environment:
~~~
python3 -m venv ./venv
. /venv/bin/activate
~~~
- Install project requirements:
~~~
pip install -r requirements.txt
~~~
Note: you may have to install the libraries torch and torchvision separately. Follow the pip instructions [here](https://pytorch.org/get-started/locally/).

## Usage
You can generate explanation maps with the pretrained attention module:
~~~
cd "TAME/scripts/bash scripts" 
. get_mask.sh <image_name> <label>
~~~
The image will be searched in the images directory. `image_name` should contain the file extension as well. If no `image_name` and `label` are provided, runs default test with image `162_166.JPEG` and label `162`.

## Training
- To train TAME on VGG-16 or ResNet-50 from scratch, run:
~~~
cd "scripts/bash scripts"
. job.sh 
~~~

**OR**, for the ResNet-50 backbone with the selected loss function:
~~~
cd scripts
sh ResNet50_train.sh
~~~
Before running any of the .sh files, set the img_dir, snapshot_dir and arch parameters inside the .sh file. For the *_CE.sh files the arch parameter must be set only with model file's names (*/L-CAM/models) with the A character at the end, for all the other .sh files the arch parameter must be set with file's names (*/L-CAM/models) without the A character at the end. The produced model will be saved in the snapshots folder. 

## Evaluation of L-CAM-Fm and L-CAM-Img
- To evaluate the model, download the pretrained models that are available in this [Google Drive](https://drive.google.com/drive/folders/1Urn-e4Aj00vyo4LP0qW0S-lfM-VYdZa_?usp=sharing), and place the downloaded zip files (resnet50_V5.zip,vgg16_V5.zip) in the snapshots folder and extract them; otherwise, use your own trained model that is placed in the snapshots folder.

- Run the commands below to calculate Increase in Confidence (IC) and Average Drop (AD), if using the VGG-16 backbone:
~~~
cd scripts
sh VGG16_AD_IC.sh 
~~~

**OR**, if using the ResNet-50 backbone:
~~~
cd scripts
sh ResNet50_AD_IC.sh
~~~
Before running any of the .sh files, again set the img_dir, snapshot_dir, arch and percentage parameters inside the .sh file.

## Evaluation of the other methods
- To evaluate the methods that are used for comparison with L-CAM-Fm and L-CAM-Img, run the commands below to calculate Increase in Confidence (IC) and Average Drop (AD):
~~~
cd scripts
sh Inference_OtherMethods.sh 
~~~
Before running  the .sh file, first take the code for Grad-Cam, Grad-Cam++, Score-CAM and RISE from [ScoreCAM](https://github.com/yiskw713/ScoreCAM/blob/master/cam.py) repository and [RISE](https://github.com/eclique/RISE) repository  and save it to */L-CAM/utils/cam.py file. Than select from */L-CAM/Inference_OtherMethod the file with the method that you want to evaluate e.g. For ResNet-50 backbone and RISE method select ResNet50_rise.py from */L-CAM/Inference_OtherMethods folder/ and set it in the Inference_OtherMethods.sh file. Also, set the img_dir and percentage parameters inside the .sh file.
For example:
~~~
CUDA_VISIBLE_DEVICES=0 python ResNet_rise.py \
--img_dir='/ssd/imagenet-1k/ILSVRC2012_img_val' \
--percentage=0.5 \
~~~

## Parameters
During the training and evaluation stages the above parameters can be specified.

|   Parameter name | Description                                                                                              | Type  |          Default Value          |
|-----------------:|:---------------------------------------------------------------------------------------------------------|:-----:|:-------------------------------:|
|     `--root_dir` | Root directory for the project.                                                                          |  str  |            ROOT_DIR             |
|      `--img_dir` | Directory where the training images reside.                                                              |  str  |             img_dir             |
|   `--train_list` | The path where the annotations for training reside.                                                      |  str  |           train_list            |
|    `--test_list` | The path where the annotations for evaluation reside.                                                    |  str  |            test_list            |
|   `--batch_size` | Selected batch size.                                                                                     |  int  |           Batch_size            |
|   `--input_size` | Image scaling parameter: the small side of each image is resized, to become equal to this many pixels.   |  int  |               256               |
|    `--crop_size` | Image cropping parameter: each (scaled) image is cropped to a square crop_size X crop_size pixels image. |  int  |               224               |
|         `--arch` | Architecture selected from the architectures that are avaiable in the models folder.                     |  str  | e.g. ResNet50_aux_ResNet18_TEST |
|           `--lr` | The initial learning rate.                                                                               | float |               LR                |
|        `--epoch` | Number of epochs used in training process.                                                               |  int  |              EPOCH              |
| `--snapshot_dir` | Directory where the trained models are stored.                                                           |  str  |          Snapshot_dir           |
|   `--percentage` | Percentage of saliency's muted pixels.                                                                   | float |             percent             |

The above parameters can be changed in the .sh files. For example:
~~~
CUDA_VISIBLE_DEVICES=0 python Evaluation_L_CAM_ResNet50.py \
	--img_dir='/m2/ILSVRC2012_img_val' \
	--snapshot_dir='/m2/gkartzoni/L-CAM/snapshots/ResNet50_L_CAM_Img' \
	--arch='ResNet50_L_CAM_Img' \
	--percentage=0 \
~~~
We use relative paths for train_list and test_list so they are specified relative to the project path (/L-CAM) in the .py files. The paths that must be specified externally are arch(from */L-CAM/models folder), snapshot_dir, img_dir and percentage, as in the example.

## Citation
<div align="justify">
    
If you find our work, code or pretrained models, useful in your work, please cite the following publication:

I. Gkartzonika, N. Gkalelis, V. Mezaris, "Learning visual explanations for DCNN-based image classifiers using an attention mechanism", 2022, under review.
</div>

BibTeX:

```
@INPROCEEDINGS{9666088,
    author    = {Gkartzonika, Ioanna and Gkalelis, Nikolaos and Mezaris, Vasileios},
    title     = {Learning visual explanations for DCNN-based image classifiers using an attention mechanism},
    booktitle = {under review},
    month     = {},
    year      = {2022},
    pages     = {}
}
```

## License
<div align="justify">
    
Copyright (c) 2022, Ioanna Gkartzonika, Nikolaos Gkalelis, Vasileios Mezaris / CERTH-ITI. All rights reserved. This code is provided for academic, non-commercial use only. Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
</div>

## Acknowledgement
The training process is based on code released in the [DANet](https://github.com/xuehaolan/DANet) repository.

The code for the methods that are used for comparison with L-CAM-Fm and L-CAM-Img is taken from the [ScoreCAM](https://github.com/yiskw713/ScoreCAM/blob/master/cam.py) repository, except for the code for the RISE method, which is taken from the [RISE](https://github.com/eclique/RISE) repository.

<div align="justify"> This work was supported by the EU Horizon 2020 programme under grant agreements H2020-951911 AI4Media and H2020-832921 MIRROR. </div>


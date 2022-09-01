# TAME: Trainable Attention Mechanism for Explanations

This repository hosts the code and data lists for our learning-based eXplainable AI (XAI) method called TAME, for Convolutional Neural Network-based (CNN) image classifiers. Our method receives as input an image and a class label and produces as output the image regions that the CNN has focused on in order to infer this class. TAME uses an attention mechanism (AM), trained end-to-end along with the original, already-trained (frozen) CNN, to derive class activation maps from feature map sets extracted from selected layers. During training, the generated attention maps of the AM are applied to the inputs. The AM weights are updated by applying backpropagation on a multi-objective loss function to optimize the appearance of the attention maps (minimize high-frequency variation and attention mask area) and minimize the cross-entropy loss. This process forces the AM to learn the image regions responsible for the CNN’s output. Two widely-used evaluation metrics, Increase in Confidence (IC) and Average Drop (AD), are used for evaluation.
- This repository contains the code for training, evaluating and applying TAME, using VGG-16 or ResNet-50 as the pre-trained backbone network along with the Attention Mechanism and our selected loss function. There is also a guide on applying TAME to any CNN classifier.
- Instead of training, the user can also use a pretrained attention mechanism for the pretrained VGG-16 or ResNet-50 classifiers in [TAME/snapshot/](https://github.com/bmezaris/TAME/tree/main/snapshots/).
- In [TAME/datalist/ILSVRC](https://github.com/bmezaris/TAME/tree/main/datalist/ILSVRC) there are text files with annotations for training VGG-16 and ResNet-50 (VGG-16_train.txt, ResNet50_train.txt) and text files with annotations for 2000 randomly selected images to be used at the validation stage (Validation_2000.txt) and 2000 randomly selected images (exclusive of the previous 2000) for the evaluation stage (Evaluation_2000.txt) of the L-CAM methods.
- The ILSVRC 2012 dataset images should be downloaded by the user manually.
- The required packages for this project are contained in requirements.txt. This project was developed with Python 3.8

---

### Table of Content
<!-- TOC -->
- [TAME: Trainable Attention Mechanism for Explanations](#tame-trainable-attention-mechanism-for-explanations)
    - [Table of Content](#table-of-content)
  - [Initial Setup](#initial-setup)
  - [Usage](#usage)
  - [Self training](#self-training)
    - [Data preparation](#data-preparation)
    - [LR test](#lr-test)
      - [LR Parameters](#lr-parameters)
    - [Training & Evaluation](#training--evaluation)
      - [Train/Evaluation Parameters](#trainevaluation-parameters)
    - [Using a different model](#using-a-different-model)
  - [Citation](#citation)
  - [License](#license)
  - [Acknowledgement](#acknowledgement)
<!-- TOC -->

---

## Initial Setup
Make sure that you have a working git, cuda, Python 3 and pip installation before proceeding.
- Clone this repository:
~~~commandline
git clone https://github.com/bmezaris/TAME
~~~
- Go to the locally saved repository path:
~~~commandline
cd TAME
~~~
- Create and activate a virtual environment:
~~~commandline
python3 -m venv ./venv
. venv/bin/activate
~~~
- Install project requirements:
~~~commandline
pip install -r requirements.txt
~~~
> __Note__: you may have to install the libraries torch and torchvision separately. Follow the pip instructions [here](https://pytorch.org/get-started/locally/). Also, make sure you have a working matplotlib GUI backend.

## Usage
You can generate explanation maps with the pretrained VGG-16 or ResNet-50 attention module:
~~~commandline
cd "./scripts/bash scripts" 
. get_mask_(VGG-16|ResNet-50).sh <image_name> <label>
cd ../../
~~~
The image will be searched for in the `./images` directory. `image_name` should contain the file extension as well. If no `image_name` and `label` are provided, the script runs a default test with image `TAME/images/162_166.JPEG` and label `162` using the `./snapshots/vgg16_TAME` model. The output together with the input image and preprocessed input image are saved in `./images/heatmaps/<model_name>`.
> **Note**:
If you want to load a model from a different directory, add the path to the checkpoint directory on the flag `--restore-dir` along with the correct `--model` and `--layers` flag.

## Self training
To self train TAME, follow the steps below.
### Data preparation
Download [here](https://image-net.org/challenges/LSVRC/2012/) the training and evaluation images for the ILSVRC 2012 dataset, then extract folders and sub-folders and place the extracted folders (ILSVRC2012_img_train, ILSVRC2012_img_val) in the `./dataset/ILSVRC2012_img_train` and `./dataset/ILSVRC2012_img_val` folders. The folder structure of the image files should look like this:
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
If you have already downloaded the ILSVRC 2012 dataset and don't want to change its location, or create a symbolic link to `./dataset`, edit the paths in the file `./scripts/bash scripts/pc_info.sh` to the location of your dataset.

### LR test
To train TAME, you should first run a learning rate test to determine the optimal maximum learning rate.
- First, run the learning rate script for the VGG-16 or ResNet-50 classifier (Choose one of the values in the parenthesis):
```commandline
cd "scripts/bash scripts"
. lr_finder_(VGG-16|ResNet-50).sh
cd ../../
```


#### LR Parameters 
(The "Default Values" below are for the VGG-16 backbone; see lr_finder_ResNet-50.sh for the corresponding default values for the ResNet-50 backbone. To see all available VERSION choices, set `VERSION=''`)

| Parameter name | Description                                                                      | Type  | Default Value                                                                  |
|----------------|----------------------------------------------------------------------------------|-------|--------------------------------------------------------------------------------|
| IMGDIR         | The path to the training images                                                  | str   | "dataset/ILSVRC2012_img_train", taken from `./scripts/bash scripts/pc_info.sh` |
| TRAIN          | The name of the training list containing the model truth labels                  | str   | VGG16_train.txt                                                                |
| MODEL          | The name of the model                                                            | str   | vgg16                                                                          |
| LAYERS         | The layers to be used by TAME                                                    | str   | "features.16 features.23 features.30"                                          |
| WD             | The weight decay value                                                           | float | 5e-4                                                                           |
| VERSION        | The attention mechanism version to be used                                       | str   | TAME                                                                           |
| BSIZE          | The batch size to be used; pick the largest value your graphics card can support | int   | 32                                                                             |

- Second, view the loss curve computed with the learning rate test using the plot_lr tool and follow the prompts
:
```commandline
cd ./scripts
python plot_lr.py
cd ../
```
- Pick a local minimum as the max learning rate for training. If results are not satisfactory, pick a different local minimum and try again.
### Training & Evaluation
To train TAME on VGG-16 or ResNet-50 from scratch, and generate the evaluation metrics AD, IC for the 100%, 50% and 15% masks, run:
~~~commandline
cd "./scripts/bash scripts"
. train_eval_(VGG-16|ResNet-50).sh 
cd ../../
~~~

#### Train/Evaluation Parameters
(The "Default Values" below are for the VGG-16 backbone; see lr_finder_ResNet-50.sh for the corresponding default values for the ResNet-50 backbone. To see all available VERSION choices, set `VERSION=''`)

| Parameter name | Description                                                                                     | Type  | Default Value                                                                  |
|----------------|-------------------------------------------------------------------------------------------------|-------|--------------------------------------------------------------------------------|
| IMGDIR         | The path to the training images                                                                 | str   | "dataset/ILSVRC2012_img_train", taken from `./scripts/bash scripts/pc_info.sh` |
| RESTORE        | The path to the checkpoint folder you want to use                                               | str   | "", the checkpoints in `./snapshots/` are used by default                      |
| TRAIN          | The name of the training list containing the model truth labels                                 | str   | VGG16_train.txt                                                                |
| TEST           | The name of the test list containing the ground truth labels used for evaluation and validation |       |                                                                                |
| MODEL          | The name of the model                                                                           | str   | vgg16                                                                          |
| LAYERS         | The layers to be used by TAME                                                                   | str   | "features.16 features.23 features.30"                                          |
| WD             | The weight decay value                                                                          | float | 5e-4                                                                           |
| VERSION        | The attention mechanism version to be used                                                      | str   | TAME                                                                           |
| BSIZE          | The batch size to be used; pick the value that was used for the learning rate test              | int   | 32                                                                             |
| MLR            | The maximum learning rate, chosen with the learning rate test                                   | float | 1e-2                                                                           |
| VALDIR         | The path to the validation and evaluation images                                                | str   | "dataset/ILSVRC2012_img_val", taken from `./scripts/bash scripts/pc_info.sh`   |

Before running the train and eval script, set the MLR variable in the .sh file from the learning rate plot you computed.


### Using a different model
To use TAME with a different classification model, follow these steps:

- Import your model in the `model_inspector.py` script, load it to the `mdl` variable and run the script. 
- Use the exact same names as the ones in train_names for the layers you choose to use.
- Write a space delimited string with all the layer names in the `layers` variable and uncomment lines 19-20.
- Run the script again, if there are no errors, you can proceed.
- In the file `./scripts/utilities/model_prep.py`, add your model in the `models_dict` dictionary.
- You can now use your model by changing the `MODEL` and `LAYERS` variables in the `*_Generic.sh` bash scripts.

## Citation
<div align="justify">
    
If you find our TAME method, code or pretrained models useful in your work, please cite the following publication:
    
M. Ntrougkas, N. Gkalelis, V. Mezaris, "TAME: Attention Mechanism Based Feature Fusion for Generating Explanation Maps of Convolutional Neural Networks", Proc. IEEE Int. Symposium on Multimedia (ISM), Naples, Italy, Dec. 2022.
</div>

BibTeX:

```
@INPROCEEDINGS{Ntrougkas2022,
    author    = {Ntrougkas, Mariano and Gkalelis, Nikolaos and Mezaris, Vasileios},
    title     = {TAME: Attention Mechanism Based Feature Fusion for Generating Explanation Maps of Convolutional Neural Networks},
    booktitle = {Proc. IEEE Int. Symposium on Multimedia (ISM)},
    month     = {Dec.},
    year      = {2022},
    pages     = {}
}
```
<div align="justify">

You may want to also consult and, if you find it also useful, also cite our earlier work on this topic (methods L-CAM-Img, L-CAM-Fm):    
    
I. Gkartzonika, N. Gkalelis, V. Mezaris, "Learning Visual Explanations for DCNN-Based Image Classifiers Using an Attention Mechanism", Proc. ECCV 2022 Workshop on Vision with Biased or Scarce Data (VBSD), Oct. 2022.
</div>

BibTeX:

```
@INPROCEEDINGS{Gkartzonika2022,
    author    = {Gkartzonika, Ioanna and Gkalelis, Nikolaos and Mezaris, Vasileios},
    title     = {Learning Visual Explanations for DCNN-Based Image Classifiers Using an Attention Mechanism},
    booktitle = {Proc. ECCV 2022 Workshop on Vision with Biased or Scarce Data (VBSD)},
    month     = {Oct.},
    year      = {2022},
    pages     = {}
}
```

## License
<div align="justify">
    
Copyright (c) 2022, Mariano Ntrougkas, Nikolaos Gkalelis, Vasileios Mezaris / CERTH-ITI. All rights reserved. This code is provided for academic, non-commercial use only. Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
</div>

## Acknowledgement
The TAME implementation was built in part on code previously released in the [L-CAM](https://github.com/bmezaris/L-CAM) repository. 

The code for the methods that are used for comparison is taken from the [L-CAM](https://github.com/bmezaris/L-CAM) repository for L-CAM-Img, the [RISE](https://github.com/eclique/RISE) repository for RISE and the [ScoreCAM](https://github.com/yiskw713/ScoreCAM/blob/master/cam.py) repository for ScoreCAM and the remaining methods.

<div align="justify"> This work was supported by the EU Horizon 2020 programme under grant agreement H2020-101021866 CRiTERIA. </div>


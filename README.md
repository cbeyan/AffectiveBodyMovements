# Code for the paper "Modeling Multiple Temporal Scales of Full-body Movements for Emotion Classification"

[Full Paper and Supplementary Material](https://ieeexplore.ieee.org/document/??) <br />
[Video Presentation](??) <br />

## Overview
![BlockDiagram](https://github.com/cbeyan/AffectiveBodyMovements/main/figure.png)

Proposed method employs a two-branch architecture. It consists of two CNNs, each of them is composed of three
convolutional layers followed by fully connected layers. The shape of the convolutional filters are extended along the time
axis to form *3x5* rectangles. The reason for having rectangular filters is that we expect the network to learn and extract features in
the time domain rather than among successive markers. It is also important to highlight that the input image is always rectangular.
The first convolution is applied to the input image with *16 filters*. The "same" padding, which makes the size of outputs the same
as that of inputs, is used. A max pooling operation with a stride of two is performed, which reduces both *x* and *y* dimensions by half. 
The obtained result is given to the next layer after applying a ReLU function.

In the second layer *32 filters* are used while in the third layer *64 filters* are used. Such increase in the number of
filters allows us to identify more complex features in the deeper layers. After the third and final max pooling and ReLU layer, the
image size is reduced to *3x12*, but with *64 layers*. The output
is then flattened out. In this two-branch architecture, separate
convolutional layers are used before the weights are flattened out
and these weights are added together in a fully-connected layer.
Finally, another fully connected layer of dimension *4x1* is used
as the output layer such that each output value corresponds to one emotion class. 
A softmax function in the output layer determines the final emotion class for the given input image.

The input of our model at a time is a part of a data segment in
the form of two RGB images in logistic position image format (see
the following figure). The two-branches of the proposed architecture take images both having the size *M=NxK* (while K is defined by the
number of markers). Strating from an image *I* having the size
*MxK*) corresponding to a certain data chunk duration and *I_2* that
is a part of *I* corresponding to the last portion of *I* (e.g., the last
quarter), thus its size is *M=NxK*, where 1 < N <M, first, image resizing with bi-cubic interpolation is
applied to *I*, resulting in *I_00*, which is *M=N xK*. Then, *I_00* and *I* are given to the network as the inputs, simultaneously.



## Sub-directories and Files
There are four sub-directories described as follows:

### images
Contains the block diagram of proposed method, and some sample RGB images in logistic position format.
 

### FCN-Training

``FCN_Train_Main``: To train Fully Convolutional ResNet-50 model on a given dataset 

``resnet_fcn.py``: ResBet-based Fully Convolutional ResNet-50 definition 

``datageneratorRealVAD.py``: Image batch generator including segmentation mask and bounding boxes

``datageneratorTest.py``: Sequential image batch generator with only bounding box annotation

## Dependencies
* Python 3.5
* Tensorflow 1.12


## How it works
1- Use the Matlab code given to obtain RGB images in logistic position format.

2- Use the Python code supplied to train and test the method.

## Reference

C. Beyan, S. Karumuri, G. Volpe, A. Camurri and R. Niewiadomski, "Modeling Multiple Temporal Scales of Full-body Movements for Emotion Classification", 
in IEEE Transactions on Affective Computing, doi: 10.1109/TAFFC.2021.??.
```
@ARTICLE{affectiveBodyMovements,
  author={Cigdem Beyan and Sukumar Karumuri and Gualtiero Volpe and Antonio Camurri and Radoslaw Niewiadomski},
  journal={IEEE Transactions on Affective Computing}, 
  title={Modeling Multiple Temporal Scales of Full-body Movements for Emotion Classification}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TAFFC.2021.??}}

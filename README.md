# Code for the paper "Modeling Multiple Temporal Scales of Full-body Movements for Emotion Classification"

[Full Paper and Supplementary Material](https://ieeexplore.ieee.org/document/??) <br />
[Video Presentation](??) <br />

## Overview
![BlockDiagram](https://github.com/cbeyan/AffectiveBodyMovements/blob/main/FigureMain.png)

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
the form of two RGB images in logistic position image format. 
The two-branches of the proposed architecture take images both having the size *M=NxK* (while K is defined by the
number of markers). Strating from an image *I* having the size
*MxK*) corresponding to a certain data chunk duration and *I_2* that
is a part of *I* corresponding to the last portion of *I* (e.g., the last
quarter), thus its size is *M=NxK*, where 1 < N <M, first, image resizing with bi-cubic interpolation is
applied to *I*, resulting in *I_00*, which is *M=N xK*. Then, *I_00* and *I* are given to the network as the inputs, simultaneously.

## Logistic Position Image Construction
Data consisting of the *3D-positions* of e.g., 30 markers at 100 fps was
converted into RGB images, which is a common input format for
CNNs. This includes dividing the MoCap segments as identified
by the experts, which have a variable duration, into chunks of fixed
duration. Then, a chunk of data is converted into an RGB image.
Various values were tested for the duration of a single chunk while
overlapping in time was also applied.

The procedure for constructing RGB images includes bodycentred
relative normalization. The value of marker e.g., CLAV at the first frame of each chunk is taken
as a point of reference. CLAV is situated on the lower part of the chest on the xiphoid process. So in the first frame, the position of
CLAV is zero. The positions of the other markers are taken with respect to this new origin. By using
body-centred relative positioning, the range of the marker values
is reduced, thus, it is no longer required to map all the positions of
the workspace. 

An 8-bit RGB image format is used to represent the data such that the X,
Y and Z coordinates of the markers are associated with the R, G
and B layers, respectively. Markers are represented on the y-axis,
while the consecutive frames of the sequence are represented on
the x-axis. For example, a row of the R layer in the resulting image
represents the temporal evolution of the X coordinate of the marker
associated with that row. Then, logistic position is used to fit
the information in this 8-bit image format. We use a logistic function that maps the positions into the -127 to
+127 interval (see paper for the equation used).

## Sub-directories and Files
There are four sub-directories described as follows:

### images
Some sample RGB images in logistic position format with their emotion classes: angry, happy, insecure, sad.
(https://github.com/cbeyan/AffectiveBodyMovements/blob/main/SAMPLEIMAGES)

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

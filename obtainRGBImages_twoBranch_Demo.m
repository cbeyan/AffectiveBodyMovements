clc
clc
clear all
clear all

imageFormat=0; 
% we not only implemented "Logistic Logistic Position Image" Construction but
% also other strategies, to run "logistic position image", keep the value
% of imageFormat as ZERO.
segmentLength=4; %1; %0.5; % 1 second
applyRelativePositionExtraction=0; 
% 1 means apply relative position extraction will be applied, 
% 0 means data augmentation will be performed

augmentMode=1; %1 for leftright for other see the code applyImageGeneration
augmentAngle=pi/2; % 90 degree of rotation will be applied

KK=4; % KK is equals to number of tsv and txt files in which the 3d data is stored.
% we include two examples files namely, K4.txt and T4.txt.

for i=4:KK
    tsv = sprintf('T%d.tsv', i);
    txt = sprintf('K%d.txt', i);
    tsv=['MocapData\' tsv];
    txt=['MocapData\' txt];
    applyImageGeneration(tsv, txt, imageFormat, i,segmentLength,applyRelativePositionExtraction,augmentMode,augmentAngle); 
end
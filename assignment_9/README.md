<br/>
<h1 align="center">Session 8: Advanced training Concepts
<br/>
<!-- toc -->
    <br>
    
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/RajamannarAanjaram/badges/)
[![Awesome Badges](https://img.shields.io/badge/badges-awesome-green.svg)](https://github.com/RajamannarAanjaram/badges)
    <br>
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/RajamannarAanjaram/)

### Contributors

<p align="center"> <b>Team - 6</b> <p>
    
| <centre>Name</centre> | <centre>Mail id</centre> | 
| ------------ | ------------- |
| <centre>Amit Agarwal</centre>         | <centre>amit.pinaki@gmail.com</centre>    |
| <centre>Pranav Panday</centre>         | <centre>pranavpandey2511@gmail.com</centre>    |
| <centre>Rajamannar A K</centre>         | <centre>rajamannaraanjaram@gmail.com</centre>    |
| <centre>Sree Latha Chopparapu</centre>         | <centre>sreelathaemail@gmail.com</centre>    |\\

<!-- toc -->
    
## Problem Statement

Write a custom ResNet architecture for CIFAR10 that has the following architecture:
 - PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
 - Layer1 -
    - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
    - R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
    - Add(X, R1)
- Layer 2 -
    - Conv 3x3 [256k]
    - MaxPooling2D
    - BN
    - ReLU
- Layer 3 -
    - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
    - R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
    - Add(X, R2)
- MaxPooling with Kernel Size 4
- FC Layer 
- SoftMax
- Uses One Cycle Policy such that:
    - Total Epochs = 24
    - Max at Epoch = 5
    - LRMIN = FIND
    - LRMAX = FIND
    - NO Annihilation
- Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by - CutOut(8, 8)
- Batch size = 512
- Target Accuracy: 90% (93% for late submission or double scores). 
- NO score if your code is not modular. Your collab must be importing your GitHub package, and then just running the model. I should be able to find the custom_resnet.py model in your GitHub repo that you'd be training. 

## Model training log





## Plots
![Train and Test Charts](images/charts.png)

<hr>

## Misclassifications

Below are examples of some missclassified examples in the test set.  

![Misclassifications](images/missclassifications.png)

## Grad-CAM outputs

Below are 10 grad-cam example images for misclasified examples in the test set.  

![cam1](images/cam1.png)
![cam2](images/cam2.png)
![cam3](images/cam3.png)
![cam4](images/cam4.png)
![cam5](images/cam5.png)
![cam6](images/cam6.png)
![cam7](images/cam7.png)
![cam8](images/cam8.png)
![cam9](images/cam9.png)
![cam10](images/cam10.png)
# Tutorials of Computer Vision

This repo includes some implementations of Computer Vision algorithms using tf2+. Codes are easy to read and follow.
If you can read Chinese, I have a teaching website for studying AI models. 

All toy implementations are organised as following:

- CNN
    - [Numpy Convolution mechanism](#ConvMechanism)
    - [LeNet](#LeNet)
    - [VGG](#VGG)
    - [GoogLeNet](#GoogLeNet)
    - [ResNet](#ResNet)
    - [DenseNet](#DenseNet)
    - [SENet](#SENet)
    - [MobileNetV1](#MobileNetV1)
    - [MobileNetV2](#MobileNetV2)
    - [Xception](#Xception)
    - [ShuffleNetV1](#ShuffleNetV1)
    - [ShuffleNetV2](#ShuffleNetV2)

# Installation
```shell script
$ git clone https://github.com/MorvanZhou/Computer-Vision
$ cd Computer-Vision
$ pip install -r requirements.txt
```
# ConvMechanism
Convolution mechanism and feature map

[code](/conv_demo.py) - [gif result](https://mofanpy.com/static/results/cv/conv_mechanism.gif)

<a target="_blank" href="https://mofanpy.com/static/results/cv/conv_mechanism.gif" style="text-align: center">
<img src="https://mofanpy.com/static/results/cv/conv_mechanism.gif" height="250px" alt="net structure">
</a>

# LeNet
[Gradient-Based Learning Applied to Document Recognition](http://www.dengfanxin.cn/wp-content/uploads/2016/03/1998Lecun.pdf)

[code](/lenet.py) - [net structure](https://mofanpy.com/static/results/cv/LeNet_structure.png)

<a target="_blank" href="https://mofanpy.com/static/results/cv/LeNet_structure.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/cv/LeNet_structure.png" height="250px" alt="net structure">
</a>

# VGG
[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

Deep stacked CNN.

[code](/vgg.py) - [net structure](https://mofanpy.com/static/results/cv/VGG_structure.png)

<a target="_blank" href="https://mofanpy.com/static/results/cv/VGG_structure.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/cv/VGG_structure.png" height="250px" alt="net structure">
</a>

# GoogLeNet
[Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)

Multi kernel size to capture different local information

[code](/googlenet.py) - [net structure](https://mofanpy.com/static/results/cv/GoogleLeNet_structure.png)

<a target="_blank" href="https://mofanpy.com/static/results/cv/GoogleLeNet_structure.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/cv/GoogleLeNet_structure.png" height="250px" alt="net structure">
</a>

# ResNet
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Add residual connection for better gradients.

[code](/resnet.py) - [net structure](https://mofanpy.com/static/results/cv/ResNet_structure.png)

<a target="_blank" href="https://mofanpy.com/static/results/cv/ResNet_structure.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/cv/ResNet_structure.png" height="250px" alt="net structure">
</a>

# DenseNet
[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

Compared with resnet, it has less filter each conv, sees more previous inputs.

[code](/densenet.py) - [net structure](https://mofanpy.com/static/results/cv/DenseNet_structure.png)

<a target="_blank" href="https://mofanpy.com/static/results/cv/DenseNet_structure.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/cv/DenseNet_structure.png" height="250px" alt="net structure">
</a>

# SENet
[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

SE is a module that learns to scale each feature map, it can be plugged in many cnn block, 
larger reduction_ratio reduce parameter size in FC layers with limited accuracy drop.

[code](/senet.py) - [net structure](https://mofanpy.com/static/results/cv/SENet_structure.png)

<a target="_blank" href="https://mofanpy.com/static/results/cv/SENet_structure.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/cv/SENet_structure.png" height="250px" alt="net structure">
</a>

# MobileNetV1
[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

Decomposed classical conv to two operations (dw+pw). Small but effective cnn optimized on mobile (cpu).

[code](/mobilenetv1.py) - [net structure](https://mofanpy.com/static/results/cv/MobileNetV1_structure.png)

<a target="_blank" href="https://mofanpy.com/static/results/cv/MobileNetV1_structure.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/cv/MobileNetV1_structure.png" height="250px" alt="net structure">
</a>

# MobileNetV2
[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

MobileNet v2 is v1 with residual block and layer rearrange (residual+pw+dw+pw):

- mobilenet v1: dw > pw
- mobilenet v2: pw > dw > pw    let dw see more feature maps

[code](/mobilenetv2.py) - [net structure](https://mofanpy.com/static/results/cv/MobileNetV2_structure.png)

<a target="_blank" href="https://mofanpy.com/static/results/cv/MobileNetV2_structure.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/cv/MobileNetV2_structure.png" height="250px" alt="net structure">
</a>

# Xception
[Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

Just like MobileNetV2 without last pw (residual+pw+dw).

[code](/xception.py) - [net structure](https://mofanpy.com/static/results/cv/Xception_structure.png)

<a target="_blank" href="https://mofanpy.com/static/results/cv/Xception_structure.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/cv/Xception_structure.png" height="250px" alt="net structure">
</a>


# ShuffleNetV1
[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)

Shuffle the output from 1x1 conv, and do group conv to reduce connections and speed up computing.
But MobileNet is better in this case, this may caused by group conv cuts off some feature map communications.

[code](/shufflenetv1.py) - [net structure](https://mofanpy.com/static/results/cv/ShuffleNetV1_structure.png)

<a target="_blank" href="https://mofanpy.com/static/results/cv/ShuffleNetV1_structure.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/cv/ShuffleNetV1_structure.png" height="250px" alt="net structure">
</a>

# ShuffleNetV2
[ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)

Further reduces parameters by switching group conv with split+concat, perform shuffle at end of block. Speed up calculation.
But MobileNet is better in this case, this may caused by group conv cuts off some feature map communications.
 
[code](/shufflenetv2.py) - [net structure](https://mofanpy.com/static/results/cv/ShuffleNetV2_structure.png)

<a target="_blank" href="https://mofanpy.com/static/results/cv/ShuffleNetV2_structure.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/cv/ShuffleNetV2_structure.png" height="250px" alt="net structure">
</a>
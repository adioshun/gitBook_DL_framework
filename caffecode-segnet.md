|제목|SegNet: A Deep Convolutional Encoder-Decoder Architecture for Semantic Pixel-Wise Labelling|
|-|-|
|코드|[alexgkendall](https://github.com/alexgkendall/caffe-segnet)|
|참고|[논문_2015](https://arxiv.org/abs/1511.00561), [홈페이지](http://mi.eng.cam.ac.uk/projects/segnet/), [Getting Started](http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html)|

# SegNet

## 1. 개요 

## 2. 설치 

[Getting Started with Docker](https://github.com/alexgkendall/SegNet-Tutorial)

```bash
cd /root
git clone git@github.com:alexgkendall/SegNet-Tutorial.git

cd /root/SegNet-Tutorial/docker 
# Docker file 이용 (#에러 )
nvidia-docker build -t caffe:gpu ./gpu  

# Dockerhub
nvidia-docker pull kmader/caffe-segnet

# 동작 확인 
nvidia-docker run -ti caffe:gpu caffe device_query -gpu 0

# get a bash in container to run examples
nvidia-docker run -ti -p 2222:2222 -p 8585:8585 --volume=$/root/docker:/root --name 'segnet' caffe:gpu /bin/bash

```




## 3. Training

## 4. Testing 

## 5. Fine tuning  






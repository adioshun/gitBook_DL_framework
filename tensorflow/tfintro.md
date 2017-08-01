# Tensorflow         

## 1. 개요 


## 2. 설치 

### 2.1 CPU



### 2.2 GPU 


### 2.3 Docker 


## 3. Tensorflow Object Detection API 

### 3.1 개요
> 출처 : [GOOD to GREAT](http://goodtogreate.tistory.com/entry/Tensorflow-Object-Detection-API-SSD-FasterRCNN)


2017.06.15월 공개 

지원 모델 
- Single Shot Multibox Detector (SSD) with MobileNet
- SSD with Inception V2
- Region-Based Fully Convolutional Networks (R-FCN) with ResNet 101
- Faster R-CNN with Resnet 101
- Faster RCNN with Inception Resnet v2

### 3.2 설치 (ubunutu 16.4, python3)

패키지 설치 
```bash
sudo apt-get install protobuf-compiler python-pil python-lxml 
sudo pip install matplotlib pillow lxml 

```

소스 다운로드 
```bash 
git clone https://github.com/tensorflow/models.git
```

Protobuf 컴파일 
```bash
# From tensorflow/models/
protoc object_detection/protos/*.proto --python_out=.
```

Add Libraries to PYTHONPATH : slim 디렉터리를 append시키기 위함이다.
```bash
# From tensorflow/models/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

설치 확인 
```bash 
python object_detection/builders/model_builder_test.py

.......
----------------------------------------------------------------------
Ran 7 tests in 0.013s

OK
```

> 튜토리얼 : `./object detection/object_detection_tutorial.ipyn`




# Tensorflow Object Detection API 

# 1. 개요
> 출처 : [GOOD to GREAT](http://goodtogreate.tistory.com/entry/Tensorflow-Object-Detection-API-SSD-FasterRCNN)


2017.06.15월 공개 

지원 모델 
- Single Shot Multibox Detector (SSD) with MobileNet
- SSD with Inception V2
- Region-Based Fully Convolutional Networks (R-FCN) with ResNet 101
- Faster R-CNN with Resnet 101
- Faster RCNN with Inception Resnet v2

# 2. 설치 

>ubunutu 16.4, python3, tf 1.2

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

## 3. 실행 

> 튜토리얼 : [./object detection/object_detection_tutorial.ipynb](https://github.com/tensorflow/models/blob/master/object_detection/object_detection_tutorial.ipynb)

> 참고 : [How to train your own Object Detector with TensorFlow’s Object Detector API](https://medium.com/towards-data-science/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)



###### [참고] 오류 

- TypeError: a bytes-like object is required, not 'str'
    - [TF 1.2이상으로 업그레이드](https://github.com/datitran/Object-Detector-App/issues/2)

export_inference_graph.py 파일 실행시 필요한 parameter 의 이름이 바뀌었습니다. 
- checkpoint_path -> trained_checkpoint_prefix 
- inference_graph_path -> output_directory 
- `python object_detection/export_inference_graph \` 가 아니라 `python object_detection/export_inference_graph.py \` 입니다.



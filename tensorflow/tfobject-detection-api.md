# Tensorflow Object Detection API 

- [Basic Tutorial](https://github.com/tensorflow/models/blob/master/object_detection/object_detection_tutorial.ipynb): Jupyter, 학습된 모델을 Load하여 이미지 내 물체 예측 

- [Quick Start: Distributed Training on the Oxford-IIIT Pets Dataset on Google Cloud](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/running_pets.md):   object detector 학습 방법 소개, **Transfer Learning**

- [Configuring the Object Detection Training Pipeline](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/configuring_jobs.md): 학습 관련 설정 변경 


# 1. 개요
> 출처 : [홈페이지](https://github.com/tensorflow/models/tree/master/object_detection), [GOOD to GREAT](http://goodtogreate.tistory.com/entry/Tensorflow-Object-Detection-API-SSD-FasterRCNN)


2017.06.15월 공개 

지원 모델 
- Single Shot Multibox Detector (SSD) with MobileNet
- SSD with Inception V2
- Region-Based Fully Convolutional Networks (R-FCN) with ResNet 101
- Faster R-CNN with Resnet 101
- Faster RCNN with Inception Resnet v2

# 2. 설치 

[설치 방법](https://github.com/adioshun/Blog_Jekyll/blob/master/2017-08-08-TF%20Object%20Detection%20API_Installation.md)

## 3. 테스트 

> 튜토리얼 : [./object detection/object_detection_tutorial.ipynb](https://github.com/tensorflow/models/blob/master/object_detection/object_detection_tutorial.ipynb)

> 참고 : [How to train your own Object Detector with TensorFlow’s Object Detector API](https://medium.com/towards-data-science/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)



###### [참고] 오류 

- TypeError: a bytes-like object is required, not 'str'
    - [TF 1.2이상으로 업그레이드](https://github.com/datitran/Object-Detector-App/issues/2)

export_inference_graph.py 파일 실행시 필요한 parameter 의 이름이 바뀌었습니다. 
- checkpoint_path -> trained_checkpoint_prefix 
- inference_graph_path -> output_directory 
- `python object_detection/export_inference_graph \` 가 아니라 `python object_detection/export_inference_graph.py \` 입니다.



## 4. 학습 


---
수정 
http://35.196.214.92:8585/notebooks/models/object_detection/object_detection_tutorial.ipynb#
```
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'frame{}.jpg'.format(i)) for i in range(1, 9999) ]
#plt.imshow(image_np)
plt.savefig('./save/{}.png'.format(image_path))
#plt.savefig('./save/plot.png')
```

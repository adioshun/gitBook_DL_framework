# Tensorflow Object Detection API 

- ~~[Basic Tutorial](https://github.com/tensorflow/models/blob/master/object_detection/object_detection_tutorial.ipynb)~~: Jupyter, 학습된 모델을 Load하여 이미지 내 물체 예측 

    - ~~[Real-Time Object Recognition App with Tensorflow and OpenCV](https://github.com/datitran/Object-Detector-App)~~: 최적화 코드 


- [Quick Start: Distributed Training on the Oxford-IIIT Pets Dataset on Google Cloud](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/running_pets.md):   object detector 학습 방법 소개, **Transfer Learning**
    
    - ~~[How to train your own Object Detector with TensorFlow’s Object Detector API](https://medium.com/towards-data-science/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)~~

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





###### [참고] 오류 

- TypeError: a bytes-like object is required, not 'str'
    - [TF 1.2이상으로 업그레이드](https://github.com/datitran/Object-Detector-App/issues/2)

export_inference_graph.py 파일 실행시 필요한 parameter 의 이름이 바뀌었습니다. 
- checkpoint_path -> trained_checkpoint_prefix 
- inference_graph_path -> output_directory 
- `python object_detection/export_inference_graph \` 가 아니라 `python object_detection/export_inference_graph.py \` 입니다.



## 4. 학습 


## 5. 다른 데이터 이용하여 학습 하기 

> [How to train your own Object Detector with TensorFlow’s Object Detector API](https://medium.com/towards-data-science/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)


### 5.1 데이터 준비 

TFRecord을 입력으로 사용함 (eg.  PASCAL VOC datasetZ)

    - images.tar.gz : 이미지(JPG, PNG)
    
    - annotations.tar.gz : LIST(X_min, Y_min, X_max, Y_max) + (Label)
    
![](http://i.imgur.com/HfGjktp.png)

    
###### Step 1. 이미지 준비 

      
###### Step 2. 수작업으로 라벨링 진행 

- [[LanelImg]](https://github.com/tzutalin/labelImg)라는 이미지 라벨링 툴을 이용

- PASCAL형태의 XML로 저장 

###### Step 3. Convert Tools 이용 TFRecord 변경 
                
- [`create_pascal_tf_record.py`](https://github.com/tensorflow/models/blob/master/object_detection/create_pascal_tf_record.py)

``` bash
# From tensorflow/models/
python object_detection/create_pet_tf_record.py \
    --label_map_path=object_detection/data/pet_label_map.pbtxt \
    --data_dir=`pwd` \
    --output_dir=`pwd`
```

    
    
- 참고 : Raccoon 이미지를 변환 하는 [3rd party](https://github.com/datitran/raccoon-dataset) 스크립트(XML - CSV - TFRecord) 

 
###### Step 4. 작업 위치로 이동 
    
        
- 저장 위치 : `tensorflow/models`
```
- images.tar.gz
- annotations.tar.gz
+ images/
+ annotations/
+ object_detection/
... other files and directories
```
    
> 이미지 크기는 300~500 pixels추천(???) -> OOM문제 발생, Batch-size조절로 가능 

###### 참고: Labeling Tools , Service 
|구분|이름|특징|
|-|-|-|
|Tool|[LanelImg](https://github.com/tzutalin/labelImg)||
|Tool|[FIAT (Fast Image Data Annotation Tool)](https://github.com/christopher5106/FastAnnotationTool)||
|Service|[CrowdFlower](https://www.crowdflower.com/)||
|Service|[CrowdAI ](https://crowdai.com/)||
|Service|[Amazon’s Mechanical Turk](https://www.mturk.com/mturk/welcome)||


### 5.2 Training Config 파일 수정 
- num_class : eg. 클래스가 하나 이면 1
- PATH : Train data PATH, Test data PATH, label map PATH
    - label map : *.pbtxt파일, id + name 으로 구성 (중요 : id는 항상 1부터 시작)
- eg. [Sample config](https://github.com/tensorflow/models/tree/master/object_detection/samples/configs), [Sample label map](https://github.com/tensorflow/models/tree/master/object_detection/data)

> 상세 설명 : [Configuring the Object Detection Training Pipeline](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/configuring_jobs.md)

### 5.3 Train 

#### A. Local 학습 

- [Running Locally](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/running_locally.md)

#### B. Cloud 학습 

- [Running on Google Cloud Platform](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/running_on_cloud.md)

### 5.4 Export Model 

- Script 이용 : [Exporting a trained model for inference](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/exporting_models.md) 

---
수정 
http://35.196.214.92:8585/notebooks/models/object_detection/object_detection_tutorial.ipynb#
```
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'frame{}.jpg'.format(i)) for i in range(1, 9999) ]
#plt.imshow(image_np)
plt.savefig('./save/{}.png'.format(image_path))
#plt.savefig('./save/plot.png')
```

# YOLO ROS: Real-Time Object Detection for ROS


> [Github](https://github.com/leggedrobotics/darknet_ros), [설치 설명](https://demura.net/misc/14494.html)


## 1. 개요 



## 2. 설치 

```python 
$ cd ~/catkin_ws/src
$ git clone --recursive https://github.com/leggedrobotics/darknet_ros.git
$ cd ..
$ catkin_make  #OR catkin_make -DCMAKE_BUILD_TYPE = Release

```



학습 값(`wrs_10000.weights`)과 설정 파일(`wrs_test.cfg`) 위치 확인 

```
$ catkin_ws / src / darknet_ros / darknet_ros / yolo_network_config / weights /
$ catkin_ws / src / darknet_ros / darknet_ros / yolo_network_config / cfg /
```


---

weight 다운로드 

```python 
# ~/catkin_ws/src/darknet_ros/darknet_ros/yolo_network_config/weights$ cat how_to_download_weights.txt


COCO data set (Yolo v2):
  wget http://pjreddie.com/media/files/yolov2.weights
  wget http://pjreddie.com/media/files/yolov2-tiny.weights

VOC data set (Yolo v2):
  wget http://pjreddie.com/media/files/yolov2-voc.weights
  wget http://pjreddie.com/media/files/yolov2-tiny-voc.weights

Yolo v3:
  wget http://pjreddie.com/media/files/yolov3.weights
  wget http://pjreddie.com/media/files/yolov3-voc.weights

# 저장 위치 : cd catkin_workspace/src/darknet_ros/darknet_ros/yolo_network_config/weights/


```
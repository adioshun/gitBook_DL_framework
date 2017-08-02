> 출처 : [Faster-R-CNN Install on Ubuntu 16.04(GTX1080 CUDA 8.0,cuDNN 5.1)](http://goodtogreate.tistory.com/entry/FasterRCNN-Install-on-Ubuntu-1604GTX1080-CUDA-80cuDNN-51)

# Caffe Faster R CNN Training

## 1. 개요 

2015년에 Microsoft에서 Caffe를 포함하여 Faster-R-CNN 소스코드 배포 
- 추가 적으로 caffe를 설치 할 필요 없음 
- 버클리대 공식 caffe에는 RoI pooling이 구현 되어 있지 않음 
    
###### 테스트 환경 
- ubuntu 16.04
- 그래픽카드는 GTX 1080이며 CUDA 8.0과 cuDNN 5.1을 사용한다.

###### 설치 
- Google Cloud , Docker

sudo docker pull jimmyli/faster-rcnn-gpu
sudo nvidia-docker run -i -t --name jimmyli2 jimmyli/faster-rcnn-gpu:latest /bin/bash




###### 설치 확인 
- Download pre-computed Faster R-CNN detectors : `./data/scripts/fetch_faster_rcnn_models.sh`
- cd $FRCN_ROOT : `./tools/demo.py`

> fetch_faster_rcnn_models.sh 수정 : `URL=https://dl.dropboxusercontent.com/s/o6ii098bu51d139/$FILE`
> /home/py-faster-rcnn/lib/fast_rcnn/config.py : set __C.USE_GPU_NMS = False


## 2. 학습 



http://goodtogreate.tistory.com/entry/Faster-R-CNN-Training

## 3. 테스트
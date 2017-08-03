> 출처 : [Faster-R-CNN Install on Ubuntu 16.04\(GTX1080 CUDA 8.0,cuDNN 5.1\)](http://goodtogreate.tistory.com/entry/FasterRCNN-Install-on-Ubuntu-1604GTX1080-CUDA-80cuDNN-51)

# Caffe Faster R CNN Training

## 1. 개요

2015년에 Microsoft에서 Caffe를 포함하여 Faster-R-CNN 소스코드 배포

* 추가 적으로 caffe를 설치 할 필요 없음 
* 버클리대 공식 caffe에는 RoI pooling이 구현 되어 있지 않음 

###### 테스트 환경 (ubuntu 16.04)
* 그래픽카드는 GTX 1080이며 CUDA 8.0과 cuDNN 5.1을 사용한다.

###### 설치 (Google Cloud , Docker)

```
sudo docker pull jimmyli/faster-rcnn-gpu
sudo nvidia-docker run -i -t --name jimmyli2 jimmyli/faster-rcnn-gpu:latest /bin/bash
sudo nvidia-docker run -i -t -p 2222:22 -p 8585:8080 --volume /home/hjlim99/docker:/root --name 'rcnn' jimmyli/faster-rcnn-gpu:latest /bin/bash
```
```
sudo docker pull tshrjn/py-faster-rcnn-demo
sudo nvidia-docker run -i -t --name tshrjn tshrjn/py-faster-rcnn-demo:latest /bin/bash
```

###### 설치 확인

* Download pre-computed Faster R-CNN detectors : `./data/scripts/fetch_faster_rcnn_models.sh`
* cd $FRCN\_ROOT : `./tools/demo.py`

> fetch\_faster\_rcnn\_models.sh 수정 : `URL=https://dl.dropboxusercontent.com/s/o6ii098bu51d139/$FILE`  
> /home/py-faster-rcnn/lib/fast\_rcnn/config.py : set \_\_C.USE\_GPU\_NMS = False


## 2. 학습

> [Good to Great](http://goodtogreate.tistory.com/entry/Faster-R-CNN-Training)

학습 데이터 다운 받기 \(VOC-2007 데이터\)

```bash
if [ ! -d data ]; then mkdir data; fi; cd data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
rm -rf *.tar; cd ../
```

pre Train model 다운 로드

```bash
$ ./data/scripts/fetch_imagenet_models.sh

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/faster_rcnn_end2end/solver.prototxt \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}
```

###### solver.prototxt
```

train_net: "models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_end2end/train.prototxt"
base_lr: 0.001
lr_policy: "step"
gamma: 0.1
stepsize: 50000
display: 20
average_loss: 100
momentum: 0.9
weight_decay: 0.0005
# We disable standard caffe solver snapshotting and implement our own snapshot
# function
snapshot: 0
# We still use the snapshot prefix, though
snapshot_prefix: "vgg_cnn_m_1024_faster_rcnn"
```


###### train.prototxt

> 모델구조

```
name: "VGG_CNN_M_1024"
layer {
  name: 'input-data'
  type: 'Python'
  top: 'data'
  top: 'im_info'
  top: 'gt_boxes'
  python_param {
    module: 'roi_data_layer.layer'
    layer: 'RoIDataLayer'
    param_str: "'num_classes': 21"
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  convolution_param {
    num_output: 96
    kernel_size: 7
    stride: 2
  }
}

```

## 3. 테스트
```
./tools/test_net.py \ 
--gpu 0 \
--def models/pascal_voc/ZF/faster_rcnn_end2end/test.prototxt \
--net ./data/faster_rcnn_models/ZF_faster_rcnn_final.caffemodel \
--imdb voc_2007_test \
--cfg experiments/cfgs/faster_rcnn_end2end.yml
```

###### test.prototxt

```
name: "VGG_CNN_M_1024"
input: "data"
input_shape {
  dim: 1
  dim: 3
  dim: 224
  dim: 224
}
input: "im_info"
input_shape {
  dim: 1
  dim: 3
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  convolution_param {
    num_output: 96
    kernel_size: 7
    stride: 2
  }
}
```

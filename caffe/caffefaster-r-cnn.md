|제목|Faster R CNN |
|-|-|
|코드|[rbgirshick](https://github.com/rbgirshick/py-faster-rcnn)|
|참고||

> 출처 : [Faster-R-CNN Install on Ubuntu 16.04\(GTX1080 CUDA 8.0,cuDNN 5.1\)](http://goodtogreate.tistory.com/entry/FasterRCNN-Install-on-Ubuntu-1604GTX1080-CUDA-80cuDNN-51)

# Caffe Faster R CNN Training

## 1. 개요

2015년에 Microsoft에서 Caffe를 포함하여 Faster-R-CNN 소스코드 배포

* 추가 적으로 caffe를 설치 할 필요 없음 
* 버클리대 공식 caffe에는 RoI pooling이 구현 되어 있지 않음 

###### 테스트 환경 (ubuntu 16.04)
* 그래픽카드는 GTX 1080이며 CUDA 8.0과 cuDNN 5.1을 사용한다.

## 2. 설치 (Google Cloud , Docker)

```bash
docker pull adioshun/faster-rcnn:20170808r1

sudo nvidia-docker run -i -t -p 2222:2222 -p 8585:8585 --volume /home/hjlim99/docker:/root --name 'rcnn2' adioshun/faster-rcnn:latest /bin/bash

```


###### 설치 확인

* Download pre-computed Faster R-CNN detectors : `./data/scripts/fetch_faster_rcnn_models.sh`
* cd $FRCN\_ROOT : `./tools/demo.py`

> fetch\_faster\_rcnn\_models.sh 수정 : `URL=https://dl.dropboxusercontent.com/s/o6ii098bu51d139/$FILE`  
> /home/py-faster-rcnn/lib/fast\_rcnn/config.py : set \_\_C.USE\_GPU\_NMS = False


## 2. 학습

> [Good to Great](http://goodtogreate.tistory.com/entry/Faster-R-CNN-Training)

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



## 3. 테스트
```
./tools/test_net.py \ 
--gpu 0 \
--def models/pascal_voc/ZF/faster_rcnn_end2end/test.prototxt \
--net ./data/faster_rcnn_models/ZF_faster_rcnn_final.caffemodel \
--imdb voc_2007_test \
--cfg experiments/cfgs/faster_rcnn_end2end.yml
```


## 4. FineTuning 

> [Use Faster RCNN and ResNet codes for object detection and image classification with your own training data](https://realwecan.blogspot.com/2016/11/use-faster-rcnn-and-resnet-codes-for.html)

---

# 입력 동영상 변경 

/workspace/py-faster-rcnn/tools/demo.py 

```python
    plt.savefig('demo_results/'+image_name)
    plt.close('all')
```

```python 
PATH_TO_TEST_IMAGES_DIR = ''
im_names = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'frame{}.jpg'.format(i)) for i in range(1, 11561) ]

#im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
#            '001763.jpg', '004545.jpg']
for im_name in im_names:
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Demo for {}'.format(im_name)
    demo(net, im_name)

```

입력 이미지 저장 위치 :`/workspace/py-faster-rcnn/data/demo`
저장 폴더 미리 생성 : `/workspace/py-faster-rcnn/tools/demo_results/`

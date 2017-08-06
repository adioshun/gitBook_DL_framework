# caffe

## 1. 개요

> 출처 : Caffe 실습, SNU 패턴 인식 및 컴퓨터 지능 연구실, 박성헌, 황지혜, 유재영

* Caffe : Convolutional Architecture for Fast Feature Embedding
* Developed by the Berkeley Vision and Learning Center \(BVLC\)
* Yangqing Jia, Evan Shelhamer, Travor Darrell

### 1.1 특징

* Pure C++/CUDA architecture
* Command line, Python, MATLAB interfaces
* Fast, well-tested code
* Pre-processing and deployment tools, reference models and examples
* Image data management
* Seamless GPU acceleration
* Large community of contributors to the open-source project

### 1.2 기능

Data pre-processing and management : `$CAFFE_ROOT/build/tools`

* Conversion from CSV and Images to LMDB 

#### A. Data ingest formats

* LevelDB or LMDB database
* In-memory \(C++ and Python only\)
* HDF5
* Image files

#### B. Pre-processing tools

* LevelDB/LMDB creation from raw images
* Training and validation set creation with shuffling
* Mean-image generation

#### C. Data transformations\(`tools.data_augmentation`\)

* Image cropping, resizing, scaling and mirroring
* Mean subtraction

### 1.3 이미지 처리

Caffe expects the images \(i.e. the dataset\) to be stored as blob of size \(N, C, H, W\)

* N being the dataset size
* C the number of channels
* H the height of the images 
* W the width of the images. 

### 1.4 LMDB I/O and Pre-processing

데이터를 LMDB에 넣어 처리 하는것을 선호

* import lmdb : 

## 2. 설치

* 현재\('17.03월\) python2 만 지원

* WITH\_PYTHON\_LAYER=1 option설치 필요

* .bashrc 설정 필요

```bash
export OPENBLAS_NUM_THREADS=(4)
export CAFFE_ROOT=/home/david/caffe
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/home/david/caffe/python:$PYTHONPATH
```

참고 : [caffe-installation](https://github.com/adioshun/Blog_Jekyll/blob/master/2017-07-18_caffe_Installation.md)

* 스크립트 이용하여 설치 : [CPU Only, Ubuntu 14.04](https://github.com/davidstutz/caffe-tools/blob/master/install_caffe.sh)


---

- ~~[Caffe 실습](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=16&ved=0ahUKEwim-Imx6ZPVAhUK_IMKHdd5DoE4ChAWCEUwBQ&url=http%3A%2F%2Fwww.osia.or.kr%2Fboard%2Finclude%2Fdownload.php%3Fno%3D63%26db%3Ddata2%26fileno%3D2&usg=AFQjCNFiJIxJd9alitUREY5NdyuFqVc6Yw)~~: [추천_pdf] 서울대학교 융합과학기술대학원, 패턴 인식 및 컴퓨터 지능 연구실

- ~~[Caffe BVLC tutorial slide](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.gc2fcdcce7_216_0)~~

- [Fast R-CNN Object detection with Caffe](http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf): pdf

- [Fully Convilutional Layer-pixelwise prediction](http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-pixels.pdf): pdf

- [CS231n](https://github.com/adioshun/gitBook_DL_framework/blob/master/caffe/CS231n%20Caffe%20Tutorial.pdf)


https://youtu.be/Qynt-TxAPOs

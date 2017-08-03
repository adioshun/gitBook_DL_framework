# caffe 



## 1. 개요 

> 출처 : Caffe 실습, SNU 패턴 인식 및 컴퓨터 지능 연구실, 박성헌, 황지혜, 유재영

- Caffe : Convolutional Architecture for Fast Feature Embedding
- Developed by the Berkeley Vision and Learning Center (BVLC)
- Yangqing Jia, Evan Shelhamer, Travor Darrell

### 1.1 특징 
- Pure C++/CUDA architecture
- Command line, Python, MATLAB interfaces
- Fast, well-tested code
- Pre-processing and deployment tools, reference models and examples
- Image data management
- Seamless GPU acceleration
- Large community of contributors to the open-source project

### 1.2 기능 
Data pre-processing and management : `$CAFFE_ROOT/build/tools`

#### A. Data ingest formats
- LevelDB or LMDB database
- In-memory (C++ and Python only)
- HDF5
- Image files

#### B. Pre-processing tools
- LevelDB/LMDB creation from raw images
- Training and validation set creation with shuffling
- Mean-image generation

#### C. Data transformations
Image cropping, resizing, scaling and mirroring
Mean subtraction



## 2. 설치 

참고 : [caffe-installation](https://github.com/adioshun/Blog_Jekyll/blob/master/2017-07-18_caffe_Installation.md)




## 3. 시각화 (NVIDIA DIGITS)


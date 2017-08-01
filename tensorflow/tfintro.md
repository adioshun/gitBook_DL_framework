# Tensorflow         

## 1. 개요 


## 2. 설치 

### 2.1 CPU
#### A. pip설치 
```bash
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp34-cp34m-linux_x86_64.whl
pip3 install --ignore-installed --upgrade $TF_BINARY_URL
```

#### B. Conda 설치 
```bash
conda install -c conda-forge tensorflow
```


### 2.2 GPU 

### A. pip설치 
CUDA installation
- [외부링크참고](https://github.com/adioshun/Blog_Jekyll/blob/master/2017-07-18-CUDA_CuDNN_Installation.md)


tensorflow, opencv, keras, etc. installation

```bash
apt-get install -y libcupti-dev  #에러발생 "/sbin/ldconfig.real: /usr/lib/nvidia-375/libEGL.so.1 is not a symbolic link"
apt-get install -y python3-pip python3-dev
pip3 install tensorflow-gpu
pip3 install --upgrade pip
```

> libcudnn.so.5: cannot open shared object file 
> - ldconfig -v 로해당라이브러리위치확인
> - LD_LIBRARY_PATH="/usr/local/cuda-8.0/targets/x86_64-linux/lib
> - export LD_LIBRARY_PATH
> - echo $LD_LIBRARY_PATH


### 2.3 Docker 






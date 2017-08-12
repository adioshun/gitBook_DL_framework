|제목|Fully Convolutional Networks|
|-|-|
|코드|[wkentaro](https://github.com/wkentaro/pytorch-fcn)|
|참고||

# PyTorch Fully Convolutional Networks

## 1. 개요 

## 2. 설치 

```bash
conda create -n py2torch python=2.7 ipykernel
source activate py2torch
git clone https://github.com/wkentaro/pytorch-fcn.git
cd pytorch-fcn

conda install pytorch cuda80 torchvision -c soumith
pip install .
```

voc 데이터 다운 받기 
```
cd example/voc
./download_dataset.sh
```
-  저장 위치 {home}/data/datasets/VOC


Model 다운 받기 (caffe)

```bash 
cd {home}/data/models/caffe

#fcn16s
wget http://dl.caffe.berkeleyvision.org/fcn16s-heavy-pascal.caffemodel

#fcn32s
wget http://dl.caffe.berkeleyvision.org/fcn32s-heavy-pascal.caffemodel

#fcn8s-atonce
wget http://dl.caffe.berkeleyvision.org/fcn8s-atonce-pascal.caffemodel
#fcn8s
wget http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel
```
> `fcn8s-atonce-heavy`파일 부재로 : `cp fcn8s-heavy-pascal.caffemodel fcn8s-atonce-heavy-pascal.caffemodel`


Caffe모델 pytorch용으로 변경 하기 
```
{home}/pytorch-fcn/examples/voc/model_caffe_to_pytorch.py

```
-  저장 위치 {home}/data/models/pytorch



## 3. Training

- [VOC Example](https://github.com/wkentaro/pytorch-fcn/tree/master/examples/voc)

```bash

./train_fcn8s.py -g 0
```




## 4. Testing 

```
cd example/voc
python evaluate.py /root/data/models/pytorch/fcn8s-heavy-pascal.pth
```


## 5. Fine tuning  





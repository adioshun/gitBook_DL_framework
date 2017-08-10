|제목|YOLO2|
|-|-|
|코드|[marvis](https://github.com/marvis/pytorch-yolo2)|
|참고||

# PyTorch YOLO2

## 1. 개요 

## 2. 설치 

```bash
# pytorch 설치 
conda create -n py2torch python=2.7 ipykernel
source activate py2torch
conda install pytorch torchvision cuda80 -c soumith

#YOLO설치
git clone git@github.com:marvis/pytorch-yolo2.git
wget http://pjreddie.com/media/files/yolo.weights
python detect.py cfg/yolo.cfg yolo.weights data/dog.jpg
```

## 3. Training


## 4. Testing 


## 5. Fine tuning  
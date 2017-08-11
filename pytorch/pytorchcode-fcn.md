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



## 3. Training

- [VOC Example](https://github.com/wkentaro/pytorch-fcn/tree/master/examples/voc)
```
cd example/voc
./download_dataset.sh
./train_fcn8s.py -g 0
```




## 4. Testing 


## 5. Fine tuning  





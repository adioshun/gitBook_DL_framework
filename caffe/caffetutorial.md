## 1. 데이터 전처리

> [caffe framework 데이터 셋팅하기](http://blog.naver.com/sogangori/220461170655)

#### A. Dataset 준비하기

* 파일과 라벨을 준비후 아래와 같은 텍스트 파일 만들기

  ```
  Subfolder1/file1.JPEG 7
  Subfolder2/file2.JPEG 3
  Subfolder3/file3.JPEG 4
  ```

* convert\_imageset.bin –backend=“leveldb” –shuffle=true imageData/ imageList.txt imageData\_levelDB

  * 사용법: 실행파일.exe \[FLAGS\] ROOTFOLDER/ LISTFILE DB\_NAME
  * Label은 0부터 시작
  * Shuffle, resize 등의 옵션을 활용

#### B. Mean image 구하기

대부분의 경우 training, testing 시에 image data에서 mean image를 뺀다

* compute\_image\_mean.bin –backend=“leveldb” imageData\_levelDB mean\_imageData.binaryproto
  * 사용법: 실행파일.exe \[FLAGS\] INPUT\_DB \[OUTPUT\_FILE\]
  * LevelDB 또는 LMDB를 이용해서 만듦
  * 실행결과 binaryproto 파일이 생성됨

## 2. 설정 파일

Training/Testing을 위해 보통 두 가지 파일을 정의함

#### A. Solver 정보를 담은 파일

* Gradient update를 어떻게 시킬 것인가에 대한 정보를 담음
* learning rate, weight decay 등의 parameter가 정의됨
* Test interval, snapshot 횟수 등 정의

#### B. Network 구조 정보를 담은 파일 : 실제 CNN 구조 정의 [\[상세설명\]](http://caffe.berkeleyvision.org/tutorial/layers.html)

* Net
  * Caffe에서 CNN \(혹은 RNN 또는 일반 NN\) 네트워크는 ‘Net’이라는 구조로 정의됨
  * Net은 여러 개의 Layer 들이 연결된 구조 Directed Acyclic Graph\(DAG\) 구조만 만족하면 어떤 형태이든 training이 가능함
* Layer

  * CNN의 한 ‘층＇을 뜻함
  * Convolution을 하는 Layer, Pooling을 하는 Layer, activation function을 통과하는 layer, input data layer, Loss를 계산하는 layer 등이 있음
  * 소스코드에는 각 layer별로 Forward propagation, Backward propagation 방법이 CPU/GPU 버전별로 구현되어 있음

* Blob

  * Layer를 통과하는 데이터 덩어리
  * Image의 경우 주로 NxCxHxW 의 4차원 데이터가 사용됨 \(N : Batch size, C :Channel Size, W : width, H : height\)

> 확장자가 .prototxt로 Google [Protocol Buffers](https://developers.google.com/protocol-buffers/) 기반

## 3. 네트워크 정의 

네트워크 모델은 Train용 Test용으로 별도로 존재 한다. [[이유??]](http://blog.naver.com/sssmate1/220501116973)

```python 
def mnist_network(lmdb_path, batch_size):
    """
    Convolutional network for MNIST classification.
    
    :param lmdb_path: path to LMDB to use (train or test LMDB)
    :type lmdb_path: string
    :param batch_size: batch size to use
    :type batch_size: int
    :return: the network definition as string to write to the prototxt file
    :rtype: string
    """
        
    net = caffe.NetSpec()
        
    net.data, net.labels = caffe.layers.Data(batch_size = batch_size, 
                                             backend = caffe.params.Data.LMDB, 
                                             source = lmdb_path, 
                                             transform_param = dict(scale = 1./255), 
                                             ntop = 2)
    
    net.augmented_data = caffe.layers.Python(net.data, python_param = dict(module = 'tools.layers', layer = 'DataAugmentationMultiplicativeGaussianNoiseLayer'))
    net.augmented_labels = caffe.layers.Python(net.labels, python_param = dict(module = 'tools.layers', layer = 'DataAugmentationDoubleLabelsLayer'))
    
    net.conv1 = caffe.layers.Convolution(net.augmented_data, kernel_size = 5, num_output = 20, weight_filler = dict(type = 'xavier'))
    net.pool1 = caffe.layers.Pooling(net.conv1, kernel_size = 2, stride = 2, pool = caffe.params.Pooling.MAX)
    net.conv2 = caffe.layers.Convolution(net.pool1, kernel_size = 5, num_output = 50, weight_filler = dict(type = 'xavier'))
    net.pool2 = caffe.layers.Pooling(net.conv2, kernel_size = 2, stride = 2, pool = caffe.params.Pooling.MAX)
    net.fc1 =   caffe.layers.InnerProduct(net.pool2, num_output = 500, weight_filler = dict(type = 'xavier'))
    net.relu1 = caffe.layers.ReLU(net.fc1, in_place = True)
    net.score = caffe.layers.InnerProduct(net.relu1, num_output = 10, weight_filler = dict(type = 'xavier'))
    net.loss =  caffe.layers.SoftmaxWithLoss(net.score, net.augmented_labels)
        
    return net.to_proto()


###`.prototxt` 포맷으로 네트워크 저장 하기 
# Set train_prototxt_path, train_lmdb_path and train_batch_size accordingly.
# Do the same for the test network below.

with open(train_prototxt_path, 'w') as f:
    f.write(str(iris_network(train_lmdb_path, train_batch_size)))

with open(test_prototxt_path, 'w') as f:
    f.write(str(iris_network(test_lmdb_path, test_batch_size)))

#Custom Layer를 작성시는 force_backward 설정을 해주어야 함 
with open(train_prototxt_path, 'w') as f:
    f.write('force_backward: true\n') # For the MNIST network it is not necessary, but for illustration purposes ...
    f.write(str(mnist_network(train_lmdb_path, train_batch_size))) 

```
> [참고] Custom Layer를 작성
> - [Deep learning tutorial on Caffe technology ](http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html): 중간 부분 `Create your custom python layer` 챕터 
> - [pyCaffe Tools, Examples and Resources](http://davidstutz.de/pycaffe-tools-examples-and-resources/#deploy) : 중간 부분 `Custom Python Layers` 챕터

#### D. Deploying Networks
training/testing .prototxt를 네트워크로 Deploy하려면 아래 2 절차를 수행 하여야 함 
- Eliminating the LMDB input layer;
- Removing the loss layer.

> This transformation can be automated by `tools.prototxt.train2deploy`.


## 4. 결과 분석 

### 4.1 Compute accuracy of the model on the test data
Once solved,
```python
accuracy = 0
batch_size = solver.test_nets[0].blobs['data'].num
test_iters = int(len(Xt) / batch_size)
for i in range(test_iters):
    solver.test_nets[0].forward()
    accuracy += solver.test_nets[0].blobs['accuracy'].data
accuracy /= test_iters

print("Accuracy: {:.3f}".format(accuracy))
```
---


- [Deep learning tutorial on Caffe technology : basic commands, Python and C++ code.](http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html)

- [pyCaffe Tools, Examples and Resources](http://davidstutz.de/pycaffe-tools-examples-and-resources/#deploy)



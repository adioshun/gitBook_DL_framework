## 1. 데이터 전처리

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

## 3. 실행

### 3.1 네트워크 정의 

#### A. 첫번쨰 방법 
```python
def iris_network(lmdb_path, batch_size):
    """
    Simple network for Iris classification.
    
    :param lmdb_path: path to LMDB to use (train or test LMDB)
    :type lmdb_path: string
    :param batch_size: batch size to use
    :type batch_size: int
    :return: the network definition as string to write to the prototxt file
    :rtype: string
    """
        
    net = caffe.NetSpec()
    net.data, net.labels = caffe.layers.Data(batch_size = batch_size, backend = caffe.params.Data.LMDB, source = lmdb_path, ntop = 2)
    net.data_aug = caffe.layers.Python(net.data, python_param = dict(module = 'tools.layers', layer = 'DataAugmentationRandomMultiplicativeNoiseLayer'))
    net.labels_aug = caffe.layers.Python(net.labels,python_param = dict(module = 'tools.layers', layer = 'DataAugmentationDuplicateLabelsLayer'))
    net.fc1 = caffe.layers.InnerProduct(net.data_aug, num_output = 12,bias_filler = dict(type = 'xavier', std = 0.1),weight_filler = dict(type = 'xavier', std = 0.1))
    net.sigmoid1 = caffe.layers.Sigmoid(net.fc1)
    net.fc2 = caffe.layers.InnerProduct(net.sigmoid1, num_output = 3,bias_filler = dict(type = 'xavier', std = 0.1),weight_filler = dict(type = 'xavier', std = 0.1))
    net.score = caffe.layers.Softmax(net.fc2)
    net.loss = caffe.layers.MultinomialLogisticLoss(net.score, net.labels_aug)
        
    return net.to_proto()
```


#### B. 두번쨰 방법 
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
```

#### C. `.prototxt` 포맷으로 네트워크 저장 하기 

```python
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

#### D. Deploying Networks
training/testing .prototxt를 네트워크로 Deploy하려면 아래 2 절차를 수행 하여야 함 
- Eliminating the LMDB input layer;
- Removing the loss layer.

> This transformation can be automated by `tools.prototxt.train2deploy`.

### 3.2 Training

#### A. cmd로 실행 

```python 
# train LeNet
caffe train -solver examples/mnist/lenet_solver.prototxt

# train on GPU 2
caffe train -solver examples/mnist/lenet_solver.prototxt -gpu 2

# resume training from the half-way point snapshot
caffe train -solver lenet_solver.prototxt -snapshot lenet_iter_5000.solverstate

# fine-tune CaffeNet model weights for style recognition
caffe train -solver finetuning/solver.prototxt -weights reference_caffenet.caffemodel

```

- 학습 시는 `-solver solver.prototxt `로 solver 지정 필수 

- snapshot이용시 `-snapshot lenet_iter_5000.solverstate` 지정 필요 

- 파인 튜닝시, 모델 초기화를 위한 `-weights model.caffemodel ` 지정 필요 


#### B. Func.로 실행 : `tools.solvers`
```python
# Assuming that the solver .prototxt has already been configured including
# the corresponding training and testing network definitions (as .prototxt).
solver = caffe.SGDSolver(prototxt_solver)
 
iterations = 1000 # Depending on dataset size, batch size etc. ...
for iteration in range(iterations):
    solver.step(1) # We could also do larger steps (i.e. multiple iterations at once).
    
    # Here we could monitor the progress by testing occasionally, 
    # plotting loss, error, gradients, activations etc.

```

#### C. Solver Configuration : `tools.solvers`


#### D. Monitoring : `tools.solvers.MonitoringSolver `



### 3.3 Testing
필요한 두가지 
- a caffemodel created during training needs
- a matching deploy .prototxt definition 

Both prerequisites are fulfilled when writing regular snapshots during training and using `tools.prototxt.train2deploy` on the generated `.prototxt` network definitions 

#### A. cmd로 실행 


```python
# score the learned LeNet model on the validation set as defined in the
# model architeture lenet_train_test.prototxt
caffe test -model lenet_train_test.prototxt -weights lenet_iter_10000.caffemodel -iterations 100
```
 - iterations=100 \ #iterations 옵션만큼 iteration 수행
 - weights=weight_file.caffemodel \ # 미리 학습된 weight 파일 (.caffemodel 확장자)
 - model=net_model.prototxt  #model은 solver가 아닌 net파일을 입력으로 줘야 함


#### B. Func.로 실행 

The network can be initialized as follows:

```python
net = caffe.Net(deploy_prototxt_path, caffemodel_path, caffe.TEST)
```

The input data can then be set by reshaping the data blob:
```python
image = cv2.imread(image_path)
net.blobs['data'].reshape(1, image.shape[2], image.shape[0], image.shape[1])
```



- `caffe.Net` is the central interface for loading, configuring, and running models. -

- `caffe.Classifier` and `caffe.Detector` provide convenience interfaces for common tasks.

- `caffe.SGDSolver` exposes the solving interface.

- `caffe.io` handles input / output with preprocessing and protocol buffers.

- `caffe.draw` visualizes network architectures.

- Caffe blobs are exposed as numpy ndarrays for ease-of-use and efficiency.

###### lenet\_solver.prototxt

```python
# The train/test net protocol buffer definition 
net: "examples/mnist/lenet_train_test.prototxt" # Net 구조를 정의한 prototxt 파일

# test_iter specifies how many forward passes the test should carry out.
# In the case of MNIST, we have test batch size 100 and 100 test iterations,
# covering the full 10,000 testing images.
test_iter: 100  # Test시에 iteration 횟수. Test_iter x batch_size만큼 test를 함

# Carry out testing every 500 training iterations.
test_interval: 500 #몇번 iteration돌때마다 test를 할 것인가?

# The base learning rate, momentum and the weight decay of the network.
# type: "SGD" # Solver type
base_lr: 0.01 # Learning rate
momentum: 0.9 # momentum
weight_decay: 0.0005 # Weight decay

# The learning rate policy
lr_policy: "inv" # Learning rate 변화를 어떻게 시킬 것인가
gamma: 0.0001
power: 0.75

# Display every 100 iterations
display: 100 #Loss를 보여주는 iteration 횟수

# The maximum number of iterations
max_iter: 10000 # 총 training iteration 수

# snapshot intermediate results
snapshot: 5000 # Iteration 횟수마다 기록을 남김
snapshot_prefix: "examples/mnist/lenet" # 프리픽스.caffemodel과 프리픽스.solverstate파일이 생성됨 

# solver mode: CPU or GPU
solver_mode: GPU
```

##### lenet\_train\_test.prototxt

```python
name: "LeNet"


######################################
# 입력 데이터와 관련된 Layer (LevelDB data)
#layer {
#  name: "mnist"
#  type: "Data"
#  top: "data" #Input Layer는 top이 두개
#  top: "label"#Input Layer는 top이 두개
#  transform_param {
#    scale: 0.00390625
#    mean_file: mean_mnist.binaryproto #Mean file 빼기
#  }
#  data_param {
#    source: "examples/mnist/mnist_train_leveldb" #LevelDB 경로
#    batch_size: 64 #Batch size
#    backend: LEVELDB
#  }
#  include {
#    phase: TRAIN # Train과 test시에 쓸 데이터를 따로 지정가능
#  }
#}
######################################

######################################
# 입력 데이터와 관련된 Layer ( Image data)
# - 이미지를 변환하지 않고 바로 넣을 때 사용
# - LevelDB 또는 LMDB를 이용할 때보다 속도 면에서 약간 느림
#layer {
#  name: "mnist"
#  type: "Data"
#  top: "data" #Input Layer는 top이 두개
#  top: "label"#Input Layer는 top이 두개
#  image_data_param {
#     shuffle: true #Shuffle 여부
#     source: "examples/mnist/dataList.txt" # Image list정보가 있는 파일, 
#                                            levelDB만들때 입력으로 쓴 파일과 같은 형태
#     batch_size: 64
#  }
#  include {
#    phase: TRAIN
#  }
#  transform_param {
#    scale: 0.00390625
#    mean_file: mean_mnist.binaryproto #Mean file 빼기
#  }
#  include: { phase: TRAIN } # Train과 test시에 쓸 데이터를 따로 지정가능
#}
######################################

######################################
# 입력 데이터와 관련된 Layer (HDF5 data)
# - 영상 이외에 실수 형태의 데이터를 넣을 수 있음
#layer {
#  name: "mnist"
#  type: "Data"
#  top: "data" #Input Layer는 top이 두개
#  top: "label"#Input Layer는 top이 두개
#  hdf5_data_param {
#    source: "examples/mnist/HDF5List.txt"
#    batch_size: 64
#  }
#  transform_param {
#    scale: 0.00390625
#    mean_file: mean_mnist.binaryproto #Mean file 빼기
#  }
#  include: { phase: TRAIN } # Train과 test시에 쓸 데이터를 따로 지정가능
#}
######################################

layer {
  name: "mnist"
  type: "Data"
  top: "data" #Input Layer는 top이 두개
  top: "label"#Input Layer는 top이 두개
  include {
    phase: TRAIN
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "examples/mnist/mnist_train_lmdb" #LevelDB 경로
    batch_size: 64 #Batch size
    backend: LMDB
  }
}
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "examples/mnist/mnist_test_lmdb"
    batch_size: 100
    backend: LMDB
  }
}



#################
# Convolution Layer
# Input size (i1xi2xi3)
# Output size (o1xo2xo3)
# Filter size (f1xf2)
# 학습할 parameter 수: o3xi3xf1xf2
###############
layer {
  name: "conv1"
  type: "Convolution"  #Convolution Layer
  bottom: "data"
  top: "conv1"
  # Layer별로 Learning rate를 다르게 조정가능. 
  # Solver에서 정한 learning rate에 곱해진 값이 해당 layer의 learning rate가 됨
  param {       
    lr_mult: 1 #첫번째는 weight에 대한 learning rate    
  }
  param {
    lr_mult: 2 #두번째는 bias에 대한 learning rate
  }
  convolution_param {
    num_output: 20 # Convolution후 output으로 나오는 feature map 개수 
                   # o1 = (i1 + 2 x pad_size – f1) / stride + 1
    kernel_size: 5 # Convolution에 쓰이는 filter의 크기
    stride: 1      # Stride 설정
    #pad: 1           # Padding 설정
    weight_filler { # Weight에 대한 initialization
      type: "xavier" #Gaussian도 많이 쓰임
    }
    bias_filler { #Bias에 대한 initialization
      type: "constant"  # Constant의 경우 value를 함께 지정 가능 , Default 0
    }
  }
}
layer {
  name: "pool1"  
  type: "Pooling"  # Pooling Layer
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX       # Max, mean, stochastic 가능
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 50
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "ip1"
  type: "InnerProduct"  # Fully connected layer
  bottom: "pool2"
  top: "ip1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 500  # Fully connected layer output 뉴런 개수
    weight_filler {
      type: "xavier"  # Initialization 
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu1"  #Activation Layer
  type: "ReLU"   # RELU, sigmoid, tanH 등 가능 (RELU는 negative_slope 설정 가능)
  bottom: "ip1"
  top: "ip1"
}
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 10
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "accuracy"             # Accuracy layer : Test 시에 Accuracy를 표시하기 위해 주로 사용
  type: "Accuracy"
  bottom: "ip2"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
layer {
  name: "loss"               #Loss layer : 가장 마지막 layer로 label과 비교해서 loss를 계산함
  type: "SoftmaxWithLoss"  
  bottom: "ip2"    # Loss Layer는 bottom이 두개   
  bottom: "label"  # Loss Layer는 bottom이 두개
  top: "loss"
}
```

---
## pycaffe interface 

```python
caffe.set_device(0)

caffe.set_mode_gpu()

solver = caffe.SGDSolver('PATH/TO/THE/SOLVER.PROTOTXT')

solver.net.blobs.items()
solver.net.params.items() 

solver.test_nets[0].forward() # test net (there can be more than one)

# forward one time of minibatch of SGD
solver.step(1)

# useful for visualizing input data
imshow(solver.net.blobs['data'].data[:8, 0].transpose(1,0,2).reshape(28, 8*28), cmap='gray'); axis('off')

# useful for visualizing filter ( display 5x5 filter 4x5 tiles)
imshow(solver.net.params['conv1'].diff[:,0].reshape(4,5,5,5).transpose(0,2,1,3).reshape(4*5, 5*5), cmap='gray'; axis('off')


train_loss[it] = solver.net.blobs['loss'].data
 

# (start the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='conv1')

solver.test_nets[0].blobs['score'].data.argmax(1)
              == solver.test_nets[0].blobs['label'].data

# when using pre-trained model
imagenet_net = caffe.Net(imagenet_net_filename, weights, caffe.TEST)

or
solver.net.copy_from(pretrained_model)


imagenet_net.forward()


# if you want to check current number of iteration, 
solver.iter

# save net

net = self.solver.net
net.save(str(SAVE_PATH))
```



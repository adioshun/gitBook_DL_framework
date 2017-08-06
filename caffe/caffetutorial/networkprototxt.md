##### lenet\_train\_test.prototxt

```python
name: "LeNet"


######################################
# 입력 데이터와 관련된 Layer (LevelDB data, for data saved in a LMDB database)
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
#    backend: LEVELDB #LMDB
#  }
#  include {
#    phase: TRAIN # Train과 test시에 쓸 데이터를 따로 지정가능
#  }
#}
######################################

######################################
# 입력 데이터와 관련된 Layer ( Image data, for data in a txt file listing all the files)
# - 이미지를 변환하지 않고 바로 넣을 때 사용
# - LevelDB 또는 LMDB를 이용할 때보다 속도 면에서 약간 느림
#layer {
#  name: "mnist"
#  type: "ImagaeData"
#  top: "data" #Input Layer는 top이 두개
#  top: "label"#Input Layer는 top이 두개
#  image_data_param {
#     shuffle: true #Shuffle 여부
#     source: "examples/mnist/dataList.txt" # Image list정보가 있는 파일, 
#                                            levelDB만들때 입력으로 쓴 파일과 같은 형태
#     batch_size: 64      
#     new_height: 256
#     new_width: 256
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
#  type: "HDF5Data"
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


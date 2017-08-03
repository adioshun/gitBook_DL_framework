## 5. 실습  : MNIST

#### A. Datasets 준비 
- cd $CAFFE_ROOT
- ./data/mnist/get_mnist.sh 
	- 파일 다운로드 : t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte,  train-images-idx3-ubyte,  train-labels-idx1-ubyte
- ./examples/mnist/create_mnist.sh
	- 파일 변환/생성 : mnist_test_lmdb, mnist_train_lmdb

#### B. 실행
- ./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt

> CPU 버전일 경우 lenet_solver.prototxt 의 solver mode를 CPU로 변경

> 로그파일 : /tmp 
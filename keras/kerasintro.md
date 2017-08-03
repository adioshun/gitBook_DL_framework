# Keras with tensorflow



## 1. 개요 


케라스는 널리 사용되고 있는 티아노\(Theano\)와 요즘 많이 쓰고 있는 텐서플로우\(TensorFlow\)를 위한 딥러닝 라이브러리입니다. 케라스는 아이디어를 빨리 구현하고 실험하기 위한 목적에 포커스가 맞춰진 만큼 굉장히 간결하고 쉽게 사용할 수 있도록 파이썬으로 구현된 상위 레벨의 라이브러리입니다. 즉 내부적으론 티아노와 텐서플로우가 구동되지만 연구자는 복잡한 티아노와 텐서플로우를 알 필요는 없습니다. 케라스는 쉽게 컨볼루션 신경망, 순환 신경망 또는 이를 조합한 신경망은 물론 다중 입력 또는 다중 출력 등 다양한 연결 구성을 할 수 있습니다.

## 2. 설치 

1. Dependency 설치
   ```
   sudo apt-get install libhdf5-dev
   pip install pyyaml h5py
   ```
2. 패키지 설치
   ```
   pip install keras
   ```

   Keras 백엔드로 Tensorflow 설정
   ```
   $ mkdir -p ~/.keras
   $ echo '{"epsilon":1e-07,"floatx":"float32","backend":"tensorflow"}' > ~/.keras/keras.json
   ```

>  Python에서 Keara 사용 : `from keras import backend as K`


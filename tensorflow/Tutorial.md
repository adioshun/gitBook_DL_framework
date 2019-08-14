# Tutorial 


## Terms

###### Placeholder

일종의 자료형, 다른 텐서를 할당하는 것

placeholder의 전달 파라미터는 다음과 같다.

```python 
placeholder(
    dtype,      # 데이터 타입을 의미하며 반드시 적어주어야 한다.
    shape=None, # 입력 데이터의 형태를 의미한다. 상수 값이 될 수도 있고 다차원 배열의 정보가 들어올 수도 있다. ( 디폴트 파라미터로 None 지정 )
    name=None   # 해당 placeholder의 이름을 부여하는 것으로 적지 않아도 된다.  ( 디폴트 파라미터로 None 지정 )
)
```

##### tf.get_variable()과 tf.get_collection()

- tf.get_variable() : 텐서의 저장 공간의 주 형태인 variable을 선언하는 방법
    - `tf.Variable()`가 원래 선언 방법. 하지만, `tf.get_variable()`를 사용하는 것이 좀 더 범용적
    - 이유 : `get_variable()`은 정의된 name filed값과 동일한 텐서가 존재할 경우, 새로 만들지 않고 기존 텐서를 불러들인다.
    
```python 
def get_variable(name,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=True,
                 collections=None,     #  variable의 소속
                 caching_device=None,
                 partitioner=None,
                 validate_shape=True,
                 custom_getter=None):
# 출처: https://eyeofneedle.tistory.com/24 [Technology worth spreading
```

- tf.get_collection() :  collection은 variable의 소속
    - 목적은 해당 variable을 코드의 다른 위치에서 불러오기 위해서
    - tf.get_collection(key)가 실행되면, key의 collection에 속하는 variable들의 리스트가 리턴


- tf.get_collection()사용법 7가지 [[자세히]](https://eyeofneedle.tistory.com/24)





---











# 하이레벨 API

> 출처 : [텐서플로우 하이레벨 API](http://bcho.tistory.com/1195), [Estimator를 이용한 모델 정의 방법](http://bcho.tistory.com/1196)

- tf.contrib:  공식 텐서플로우의 하이레벨 API

- Keras :  공식 하이레벨 API로 로 편입

## 1. Estimator 

![](http://cfile30.uf.tistory.com/image/9910C53359AF8CA334DC82)

Estimator: 학습(Training), 테스트(Evaluation), 예측(Prediction)을 한후, 학습이 완료된 모델을 저장(Export)하여 배포 단계를 추상화 한것 

Estimator는 
- 직접 개발자가 모델을 직접 구현하여 Estimator를 개발할 수 도 있고 (Custom Estimator) 
- 또는 이미 텐서플로우 tf.contrib.learn에 에 미리 모델들이 구현되어 있다. 

### 1.1 Estimator 예제

https://github.com/bwcho75/tensorflowML/blob/master/HighLevel%20API%201.%20Linear%20Regression%20Estimator.ipynb


---

- [새로운 텐서플로 개발 트랜드 Estimator](http://chanacademy.tistory.com/33)



---
# Tensorflow 저장 및 불러 오기 


## 1. 모델 zoo

- [Object Detection Model Zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md)

## 2. 저장 하기 

### 2,1 모델 체크포인트 .ckpt 
- 재학습 가능 모델에대한 메타정보 포함 
- 파일 크기가 크다 
- graph.pbtxt : 노드 정보가 모두 기록, .ckpt와 같이 생성 됨, input_graph 옵션의 입력값으로 활용됨 
- `tf.train.Saver().save(sess, 'trained.ckpt')` :학습한 변수 값들을  ckpt 체크포인트로 저장


### 2.2 pb 파일
- 재학습 불가능 
- 메타 데이타는 제외하고 모델과 가중치 값 포함 (모델의 그래프 + 학습된 변수값)
- tensorflow API를 이용한 C++ 프로그램에서 사용하는 포맷
- `tf.train.write_graph(sess.graph_def, ".", 'trained.pb', as_text=False)`: 그래프 저장


###### [참고] trained.ckpt+ trained.pb -> frozen_graph.pb 변환 툴 : [freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)
- out_node_names 옵션 : 자기가 사용하고자 하는 모델의 출력 노드 지정, `graph.pbtxt`파일 참고 
- 오류시 파일 상단 설정 부분에 `--input_binary=true` 추가 


> 참고 : [The tensorgraph is a example show how to generate, load graph from tensorflow](https://github.com/JackyTung/tensorgraph)




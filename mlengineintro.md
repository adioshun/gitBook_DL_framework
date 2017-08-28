[로컬에서 쌩쌩 딥러닝 코딩하기 ML Engine](http://chanacademy.tistory.com/30)

- [Cloud ML Engine Overview](https://cloud.google.com/ml-engine/docs/concepts/technical-overview)

- [명령어 정리](https://cloud.google.com/sdk/gcloud/reference/ml-engine/)

## 1. 환경 설정 

### 1. 1 Cloud Shell

- 구글 콘솔창 우측 상단의 `웹 기반 shell`활용 

- 미리 모든 패키지가 설치되어 있음 

### 2.2 Local(MAC/Linux)

```
# installation 
export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)"
echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install google-cloud-sdk
```

### 2.3 Setup 

```
gcloud init
# login
# project setup # gcloud config set project [selected-project-id]

# 2.4 ML 모델 확인  
gcloud ml-engine models list
```


## 3. Example 

디렉토리 구조 
```
- pacakge
    - model.py
    - simple_code.py
    - _init_.py
- setup.py
- ml_engine.sh
- config.yaml

```

###### simple_code.py

```python
%writefile ./package/simple_code.py  #jupyter로 코드 작성시 자동 저장 

import tensorflow as tf

const = tf.constant("hello tensorflow")

with tf.Session() as sess:
  result = sess.run(const)
  print(result)
```

###### ml_engine.sh
![](http://i.imgur.com/MXSlHjX.png)

```bash
%bash
gcloud ml-engine jobs submit training first_job_submit \
--module-name=package.simple_code \
--package-path=$(pwd)/package \
--region=us-east1 \
--staging-bucket=gs://ml_engine \
--scale-tier=BASIC_GPU

# 추가 Argument지정 가능 
# --arg1
# --arg2

# Multi_GPU
#--scale-tier=CUSTOM
#--config=./config.yaml
```

> [scale-tier옵션](https://cloud.google.com/ml-engine/docs/concepts/training-overview)

###### __init__.py
- 일반 폴더가 아닌 패키지임을 표시하기 위해 사용
- 패키지를 초기화하는 파이썬 코드를 넣을 수 있다



###### config.yaml
```
%writefile ./config.yaml
trainingInput:
  masterType: complex_model_m_gpu #GPU4개
  masterType: complex_model_l_gpu #GPU8개


````



## Commands

```shell
# 로컬에서 ml-engine을 이용한 모델 학습,예측 
# 올리기 전에 점검용 
gcloud ml-engine local train
gcloud ml-engine local prediction

# gclod 학습,예측 Job 제출 
gcloud ml-engine submit training
gcloud ml-engine submit prediction


# 





```




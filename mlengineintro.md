[로컬에서 쌩쌩 딥러닝 코딩하기 ML Engine](http://chanacademy.tistory.com/30)

https://github.com/hunkim/GoogleCloudMLExamples


- [Cloud ML Engine Overview](https://cloud.google.com/ml-engine/docs/concepts/technical-overview)

## 1. 환경 설정 

### 1. 1 Cloud Shell

- 구글 콘솔창 우측 상단의 `웹 기반 shell`활용 

- 미리 모든 패키지가 설치되어 있음 

### 2. 2 Local(MAC/Linux)

```

curl https://storage.googleapis.com/cloud-ml/scripts/setup_cloud_shell.sh | bash
export PATH=${HOME}/.local/bin:${PATH}

# 세팅 확인 
curl https://storage.googleapis.com/cloud-ml/scripts/check_environment.py | python
```

`wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-168.0.0-linux-x86_64.tar.gz`

### 2. 3
```
# 초기화 
gcloud beta ml init-project

# 저장소 설정


```

##
디렉토리 구조 
```
- pacakge
    - model.py
    - trainer.py
    - _init_.py
- setup.py
- ml_engine.sh
```

![](http://i.imgur.com/mucTcZr.png)

![](http://i.imgur.com/MXSlHjX.png)



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
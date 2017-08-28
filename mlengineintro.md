[로컬에서 쌩쌩 딥러닝 코딩하기 ML Engine](http://chanacademy.tistory.com/30)

https://github.com/hunkim/GoogleCloudMLExamples


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
```
### 2.4 ML 모델 확인  

```
gcloud ml-engine models list
```

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
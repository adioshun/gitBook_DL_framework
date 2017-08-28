[로컬에서 쌩쌩 딥러닝 코딩하기 ML Engine](http://chanacademy.tistory.com/30)



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
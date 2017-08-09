# caffe cmd 





## 1. Training(cmd)


```python 
# train LeNet
caffe train -solver examples/mnist/lenet_solver.prototxt

# train on GPU 2
caffe train -solver examples/mnist/lenet_solver.prototxt -gpu 2
```

- 학습 시는 `-solver solver.prototxt `로 solver 지정 필수 






###### [Tip] Training을 중간에 멈춘 뒤 이어서 하고 싶을때(Resume the training)

Snapshot으로 남겨둔 solverstate파일을 이용 \(-snapshot 옵션\)

```bash
caffe train \
    --solver=solver.prototxt \
    --snapshot=caffenet_train_iter_10000.solverstate
```
- `--solver=solver.prototxt`: 옵션 solver와 solver가 작동시킬 prototxt 파일입니다.
- `--snapshot=caffenet_train_iter_10000.solverstate`: 어디서부터 시작할지를 나타내는 옵션 

원리는 solverstate를 확장자로 가지는 파일에서 weight와 iteration 관련 정보를 가져오고, 해당하는 iteration부터 다시 training을 시작합니다.





## 2. Testing(cmd)
필요한 두가지 
- a caffemodel created during training needs
- a matching deploy .prototxt definition 

Both prerequisites are fulfilled when writing regular snapshots during training and using `tools.prototxt.train2deploy` on the generated `.prototxt` network definitions 


```python
# score the learned LeNet model on the validation set as defined in the
# model architeture lenet_train_test.prototxt
caffe test \
    -model train_test.prototxt \
    -weights iter_10000.caffemodel \
    -iterations 100
```
 - iterations=100 \ #iterations 옵션만큼 iteration 수행
 - weights=weight_file.caffemodel \ # 미리 학습된 weight 파일 (.caffemodel 확장자)
 - model=net_model.prototxt  #model은 solver가 아닌 net파일을 입력으로 줘야 함



## 3. Fine tuning / Transfer learning(cmd)

> [fine-tuning.ipynb](http://nbviewer.jupyter.org/github/BVLC/caffe/blob/tutorial/examples/03-fine-tuning.ipynb)

![](http://i.imgur.com/OXAJisv.png)


### 3.1 net.prototxt  수정 
- FCN에서 분류 목적 (1,000개 분류 -> 20개 분류)따라 수정 
- Fine Truning할 Layer(FCN)의 이름 변경 : fc8 -> fc8_flickr



### 3.2 solver.prototxt 수정
, 새로 추가된 레이어는 빠르게 학습하게 하기 위해

- 기존 모델은 새 데이터에 대해 천천히 반응(바뀌고)하게 하기 위해 : 학습(`base_lr`)률 수치(`0.001`) 줄이기. 
    - `fc8_flickr`을 제외한 다른 Layer의 Finetuning을 방지 하기 위해 `lr_mult`을 `0`으로 설정 할수 있다. 



- 단, 새로 추가된 레이어의 `lr_mult`는 boost(`10`)하기 
    - lr_mult: 0/학습이 안됨
    - lr_mult: 0.1/학습거의조금
    - lr_mult: 10/학습필요하기에 많이 됨. 
    - (그 layer의 최종 learning rate는 base_lr * lr_mult와 관련됨)
    




- we set `stepsize` in the solver to a lower value than if we were training from scratch, 
    - since we’re virtually far along in training and therefore want the learning rate to go down faster

> 버젼 업되면서 `base_lr`은 `solver.protxt`에 `lr_mult`는` deploy.protxt`로 위치 바뀜 


### 3.3 Pre Trained model 다운로드 


### 3.4 weights 옵션으로 Train하기 

```python 
caffe train \
    -solver finetuning/solver.prototxt \
    -weights reference_caffenet.caffemodel
```

* weights 옵션 : Snapshot으로 남겨둔 caffemodel파일을 이용 \
    - layer의 name이 동일하면 그 weight를 가져와서 초기값으로 사용
    - layer의 name이 없다면, network에 정의된 방식으로 초기값으로 사용 

> Layer 이름을 비교해서 이름이 같은 Layer는 caffemodel파일에서 미리 training된 weight를 가져오고 새로운 layer는 새로 initialization을 해서 학습함.







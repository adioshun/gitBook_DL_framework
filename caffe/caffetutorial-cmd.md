### 3.2 Training(cmd)

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

- **파인 튜닝**시, 모델 초기화를 위한 `-weights model.caffemodel ` 지정 필요 


> [pycaffe로 fine-tuning하기(K)](http://yochin47.blogspot.com/2016/03/pycaffe-fine-tuning.html)


### 3.3 Testing(cmd)
필요한 두가지 
- a caffemodel created during training needs
- a matching deploy .prototxt definition 

Both prerequisites are fulfilled when writing regular snapshots during training and using `tools.prototxt.train2deploy` on the generated `.prototxt` network definitions 


```python
# score the learned LeNet model on the validation set as defined in the
# model architeture lenet_train_test.prototxt
caffe test -model lenet_train_test.prototxt -weights lenet_iter_10000.caffemodel -iterations 100
```
 - iterations=100 \ #iterations 옵션만큼 iteration 수행
 - weights=weight_file.caffemodel \ # 미리 학습된 weight 파일 (.caffemodel 확장자)
 - model=net_model.prototxt  #model은 solver가 아닌 net파일을 입력으로 줘야 함






## 4. Tip

#### A. Training을 중간에 멈춘 뒤 이어서 하고 싶을때
- Snapshot으로 남겨둔 solverstate파일을 이용 (-snapshot 옵션)
- caffe train –solver=solver.prototxt -snapshot=lenet_iter_5000.solverstate

#### B. Fine tuning / Transfer learning
- Pre-trained model을 이용하는 방법
- Snapshot으로 남겨둔 caffemodel파일을 이용 (-weights 옵션)
- caffe train –solver=solver.prototxt –weights=lenet_iter_5000.caffemodel
- Layer 이름을 비교해서 이름이 같은 Layer는 caffemodel파일에서 미리 training된 weight를 가져오고 새로운 layer는 새로 initialization을 해서 학습함.

![](http://i.imgur.com/OXAJisv.png)
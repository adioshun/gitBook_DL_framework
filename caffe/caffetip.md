# caffe_Tip






## 2. Fine tuning / Transfer learning

> [fine-tuning.ipynb](http://nbviewer.jupyter.org/github/BVLC/caffe/blob/tutorial/examples/03-fine-tuning.ipynb)

![](http://i.imgur.com/OXAJisv.png)


### 2.1 net.prototxt  수정 
- FCN에서 분류 목적 (1,000개 분류 -> 20개 분류)따라 수정 
- Fine Truning할 Layer(FCN)의 이름 변경 : fc8 -> fc8_flickr



### 2.2 solver.prototxt 수정
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


### 2.3 Pre Trained model 다운로드 


### 2.4 weights 옵션으로 Train하기 

```python 
caffe train \
    -solver finetuning/solver.prototxt \
    -weights reference_caffenet.caffemodel
```

* weights 옵션 : Snapshot으로 남겨둔 caffemodel파일을 이용 \
    - layer의 name이 동일하면 그 weight를 가져와서 초기값으로 사용
    - layer의 name이 없다면, network에 정의된 방식으로 초기값으로 사용 

> Layer 이름을 비교해서 이름이 같은 Layer는 caffemodel파일에서 미리 training된 weight를 가져오고 새로운 layer는 새로 initialization을 해서 학습함.




![](http://i.imgur.com/OXAJisv.png)


## 4. Visualization

```python
def visualize_kernels(net, layer, zoom = 5):
    """
    Visualize kernels in the given convolutional layer.

    :param net: caffe network
    :type net: caffe.Net
    :param layer: layer name
    :type layer: string
    :param zoom: the number of pixels (in width and height) per kernel weight
    :type zoom: int
    :return: image visualizing the kernels in a grid
    :rtype: numpy.ndarray
    """

    num_kernels = net.params[layer][0].data.shape[0]
    num_channels = net.params[layer][0].data.shape[1]
    kernel_height = net.params[layer][0].data.shape[2]
    kernel_width = net.params[layer][0].data.shape[3]

    image = numpy.zeros((num_kernels*zoom*kernel_height, num_channels*zoom*kernel_width))
    for k in range(num_kernels):
        for c in range(num_channels):
            kernel = net.params[layer][0].data[k, c, :, :]
            kernel = cv2.resize(kernel, (zoom*kernel_height, zoom*kernel_width), kernel, 0, 0, cv2.INTER_NEAREST)
            kernel = (kernel - numpy.min(kernel))/(numpy.max(kernel) - numpy.min(kernel))
            image[k*zoom*kernel_height:(k + 1)*zoom*kernel_height, c*zoom*kernel_width:(c + 1)*zoom*kernel_width] = kernel

    return image
```

> 중간 처리 과정 시각화 하기 : [Classification: Instant Recognition with Caffe](https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb), 마지막 부분 참고 

## 5. Monitoring

```python
def count_errors(scores, labels):
    """
    Utility method to count the errors given the ouput of the
    "score" layer and the labels.

    :param score: output of score layer
    :type score: numpy.ndarray
    :param labels: labels
    :type labels: numpy.ndarray
    :return: count of errors
    :rtype: int
    """

    return numpy.sum(numpy.argmax(scores, axis = 1) != labels) 

solver = caffe.SGDSolver(prototxt_solver)
callbacks = []

# Callback to report loss in console. Also automatically plots the loss
# and writes it to the given file. In order to silence the console,
# use plot_loss instead of report_loss.
report_loss = tools.solvers.PlotLossCallback(100, '/loss.png') # How often to report the loss and where to plot it
callbacks.append({
    'callback': tools.solvers.PlotLossCallback.report_loss,
    'object': report_loss,
    'interval': 1,
})

# Callback to report error in console.
# Needs to know the training set size and testing set size and
# is provided with a function count_errors to count (or calculate) the errors
# given the labels and the network output
report_error = tools.solvers.PlotErrorCallback(count_errors, training_set_size, testing_set_size, 
                                               '', # may be used for saving early stopping models, uninteresting here ... 
                                               'error.png') # where to plot the error
callbacks.append({
    'callback': tools.solvers.PlotErrorCallback.report_error,
    'object': report_error,
    'interval': 500,
})

# Callback for saving regular snapshots using the snapshot_prefix in the
# solver prototxt file.
callbacks.append({
    'callback': tools.solvers.SnapshotCallback.write_snapshot,
    'object': tools.solvers.SnapshotCallback(),
    'interval': 500,
})

monitoring_solver = tools.solvers.MonitoringSolver(solver)
monitoring_solver.register_callback(callbacks)
monitoring_solver.solve(args.iterations)
```

여러 snippets

* Get Layer Names
* Copy Weights
* Create a Snapshot
* Get Batch Size
* Get the Loss
* Compute Gradient Magnitude
* Silencing Caffe Logging

> 출처 : [pyCaffe Tools, Examples and Resources](http://davidstutz.de/pycaffe-tools-examples-and-resources/#deploy)

## 6. 기타

---


- ~~[Fine-tuning CaffeNet for Style Recognition on “Flickr Style” Data](http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html)~~: [[번역]](http://hamait.tistory.com/520)

- [pycaffe로 fine-tuning하기](http://yochin47.blogspot.com/2016/03/pycaffe-fine-tuning.html)

- [Use Faster RCNN and ResNet codes for object detection and image classification with your own training data](https://realwecan.blogspot.com/2016/11/use-faster-rcnn-and-resnet-codes-for.html)





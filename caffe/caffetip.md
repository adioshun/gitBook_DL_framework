# Tip

## 1. Training을 중간에 멈춘 뒤 이어서 하고 싶을때
- Snapshot으로 남겨둔 solverstate파일을 이용 (-snapshot 옵션)
- caffe train –solver=solver.prototxt -snapshot=lenet_iter_5000.solverstate

## 2. B. Fine tuning / Transfer learning
- Pre-trained model을 이용하는 방법
- Snapshot으로 남겨둔 caffemodel파일을 이용 (-weights 옵션)
- caffe train –solver=solver.prototxt –weights=lenet_iter_5000.caffemodel
- Layer 이름을 비교해서 이름이 같은 Layer는 caffemodel파일에서 미리 training된 weight를 가져오고 새로운 layer는 새로 initialization을 해서 학습함.

![](http://i.imgur.com/OXAJisv.png)


## 3. data LMDB에 넣기 

> [Training Multi-Layer Neural Network with Caffe](http://nbviewer.jupyter.org/github/joyofdata/joyofdata-articles/blob/master/deeplearning-with-caffe/Neural-Networks-with-Caffe-on-the-GPU.ipynb)


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

- Get Layer Names
- Copy Weights
- Create a Snapshot
- Get Batch Size
- Get the Loss
- Compute Gradient Magnitude
- Silencing Caffe Logging

> 출처 : [pyCaffe Tools, Examples and Resources](http://davidstutz.de/pycaffe-tools-examples-and-resources/#deploy)


## 6. 기타 

---


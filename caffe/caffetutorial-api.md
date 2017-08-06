### 3.2 Training

#### B. Func.로 실행 : `tools.solvers`
```python
# Assuming that the solver .prototxt has already been configured including
# the corresponding training and testing network definitions (as .prototxt).
solver = caffe.SGDSolver(prototxt_solver)
 
iterations = 1000 # Depending on dataset size, batch size etc. ...
for iteration in range(iterations):
    solver.step(1) # We could also do larger steps (i.e. multiple iterations at once).
    
    # Here we could monitor the progress by testing occasionally, 
    # plotting loss, error, gradients, activations etc.

```

#### C. Solver Configuration : `tools.solvers`


#### D. Monitoring : `tools.solvers.MonitoringSolver `



### 3.3 Testing
필요한 두가지 
- a caffemodel created during training needs
- a matching deploy .prototxt definition 

Both prerequisites are fulfilled when writing regular snapshots during training and using `tools.prototxt.train2deploy` on the generated `.prototxt` network definitions 


The network can be initialized as follows:

```python
net = caffe.Net(deploy_prototxt_path, caffemodel_path, caffe.TEST)
```

The input data can then be set by reshaping the data blob:
```python
image = cv2.imread(image_path)
net.blobs['data'].reshape(1, image.shape[2], image.shape[0], image.shape[1])
```



- `caffe.Net` is the central interface for loading, configuring, and running models. -

- `caffe.Classifier` and `caffe.Detector` provide convenience interfaces for common tasks.

- `caffe.SGDSolver` exposes the solving interface.

- `caffe.io` handles input / output with preprocessing and protocol buffers.

- `caffe.draw` visualizes network architectures.

- Caffe blobs are exposed as numpy ndarrays for ease-of-use and efficiency.

---
## pycaffe interface 

```python
caffe.set_device(0)

caffe.set_mode_gpu()

solver = caffe.SGDSolver('PATH/TO/THE/SOLVER.PROTOTXT')

solver.net.blobs.items()
solver.net.params.items() 

solver.test_nets[0].forward() # test net (there can be more than one)

# forward one time of minibatch of SGD
solver.step(1)

# useful for visualizing input data
imshow(solver.net.blobs['data'].data[:8, 0].transpose(1,0,2).reshape(28, 8*28), cmap='gray'); axis('off')

# useful for visualizing filter ( display 5x5 filter 4x5 tiles)
imshow(solver.net.params['conv1'].diff[:,0].reshape(4,5,5,5).transpose(0,2,1,3).reshape(4*5, 5*5), cmap='gray'; axis('off')


train_loss[it] = solver.net.blobs['loss'].data
 

# (start the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='conv1')

solver.test_nets[0].blobs['score'].data.argmax(1)
              == solver.test_nets[0].blobs['label'].data

# when using pre-trained model
imagenet_net = caffe.Net(imagenet_net_filename, weights, caffe.TEST)

or
solver.net.copy_from(pretrained_model)


imagenet_net.forward()


# if you want to check current number of iteration, 
solver.iter

# save net

net = self.solver.net
net.save(str(SAVE_PATH))
```



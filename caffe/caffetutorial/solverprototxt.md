###### lenet\_solver.prototxt

```python
# The train/test net protocol buffer definition 
net: "examples/mnist/lenet_train_test.prototxt" # Net 구조를 정의한 prototxt 파일

# test_iter specifies how many forward passes the test should carry out.
# In the case of MNIST, we have test batch size 100 and 100 test iterations,
# covering the full 10,000 testing images.
test_iter: 100  # Test시에 iteration 횟수. Test_iter x batch_size만큼 test를 함

# Carry out testing every 500 training iterations.
test_interval: 500 #몇번 iteration돌때마다 test를 할 것인가?

# The base learning rate, momentum and the weight decay of the network.
# type: "SGD" # Solver type
base_lr: 0.01 # Learning rate
momentum: 0.9 # momentum
weight_decay: 0.0005 # Weight decay

# The learning rate policy
lr_policy: "inv" # Learning rate 변화를 어떻게 시킬 것인가
gamma: 0.0001
power: 0.75

# Display every 100 iterations
display: 100 #Loss를 보여주는 iteration 횟수

# The maximum number of iterations
max_iter: 10000 # 총 training iteration 수

# snapshot intermediate results
snapshot: 5000 # Iteration 횟수마다 기록을 남김
snapshot_prefix: "examples/mnist/lenet" # 프리픽스.caffemodel과 프리픽스.solverstate파일이 생성됨 

# solver mode: CPU or GPU
solver_mode: GPU
```

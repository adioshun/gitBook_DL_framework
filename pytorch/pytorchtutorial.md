# PyTorch YOLO2

> [PyTorch 튜토리얼 1~10]()
> [PyTorch MNIST Example similar to TensorFlow Tutorial](https://github.com/rickiepark/pytorch-examples/blob/master/mnist.ipynb)


## 1. 개요 

## 1.1 기본 import 패키지 
```python 
import torch # arrays on GPU
import torch.autograd as autograd #build a computational graph
import torch.nn as nn ## neural net library
import torch.nn.functional as F ## most non-linearities are here
import torch.optim as optim # optimization package
```


### 1.2 파일 입력 : `DataLoader()`

```
train_loader = torch.utils.data.DataLoader()
```


## 2. Modeling

```python 
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        # input is 28x28
        # padding=2 for same padding
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        # feature map size is 14*14 by pooling
        # padding=2 for same padding
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        # feature map size is 7*7 by pooling
        self.fc1 = nn.Linear(64*7*7, 1024)
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64*7*7)   # reshape Variable
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
model = MnistModel()
```

###### [Tip] 모델내 파라미터(Weight, biase)확인 
```python
for p in model.parameters():
    print(p.size())

```

```python
params = list(model.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight


출처: http://bob3rdnewbie.tistory.com/316 [Newbie Hacker]
```

## 3. Training : `model.train()`

### 3.1 학습 알고리즘 정의 

```python 
optimizer = optim.Adam(model.parameters(), lr=0.0001)
```

### 3.2 Loss 함수 정의 : `nn.MSELoss()`
```python 
output = net(input)
target = Variable(torch.arange(1, 11))  # a dummy target, for example
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```

### 3.3 역전파 : ` loss.backward()`

loss.backward()를 호출하고 backward() 호출 이전과 이후의 바이어스 그라디언트를 살펴볼 것이다.
```python
net.zero_grad()     # zeroes the gradient buffers of all parameters
print(net.conv1.bias.grad)

loss.backward()

print(net.conv1.bias.grad)
```




###### [전체 코드] 

```python 
model.train()
train_loss = []
train_accu = []
i = 0
for epoch in range(15):
    for data, target in train_loader:
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()    # calc gradients
        train_loss.append(loss.data[0])
        optimizer.step()   # update gradients
        prediction = output.data.max(1)[1]   # first column has actual prob.
        accuracy = prediction.eq(target.data).sum()/batch_size*100
        train_accu.append(accuracy)
        if i % 1000 == 0:
            print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(i, loss.data[0], accuracy))
        i += 1
```



model.train()

## 4. Testing : `model.eval()`

model.eval() 



###### [전체 코드] 

```python
model.eval()
correct = 0
for data, target in test_loader:
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()

print('Test set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))
```
## 5. Fine tuning  




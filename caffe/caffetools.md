# caffe_Tools(int)

## 1. LMDB I/O and Pre-processing

### 1.1 how to write data to an LMDB:

```python
import tools.lmdb_io # LMDB I/O tools in caffe-tools
 
lmdb_path = 'tests/test_lmdb'
lmdb = tools.lmdb_io.LMDB(lmdb_path)
 
# Some random images in uint8:
write_images = [(numpy.random.rand(10, 10, 3)*255).astype(numpy.uint8)]*10
write_labels = [0]*10
        
lmdb.write(write_images, write_labels)
read_images, read_labels, read_keys = lmdb.read()
```

### 1.2 For reading the created LMDB
```python
def read(self):
        """
        Read the whole LMDB. The method will return the data and labels (if
        applicable) as dictionary which is indexed by the eight-digit numbers
        stored as strings.
 
        :return: read images, labels and the corresponding keys
        :rtype: ([numpy.ndarray], [int], [string])
        """
        
        images = []
        labels = []
        keys = []
        env = lmdb.open(self._lmdb_path, readonly = True)
        
        with env.begin() as transaction:
            cursor = transaction.cursor();
            
            for key, raw in cursor:
                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(raw)
                
                label = datum.label
                
                if datum.data:
                    image = numpy.fromstring(datum.data, dtype = numpy.uint8).reshape(datum.channels, datum.height, datum.width).transpose(1, 2, 0)
                else:
                    image = numpy.array(datum.float_data).astype(numpy.float).reshape(datum.channels, datum.height, datum.width).transpose(1, 2, 0)
                
                images.append(image)
                labels.append(label)
                keys.append(key)
        
        return images, labels, keys



```


### 1.3 Conversion from CSV and Images to LMDB

```python

import tools.pre_processing
import tools.lmdb_io
 
# The below example reads the CSV and writes both the data and the label
# to an LMDB. The data is normalized by the provided maximum value 7.9.
# In order to find and convert the label, its column index and a label mapping
# is provided (i.e. 'Iri-setosa' is mapped to label 0 etc.).
lmdb_converted = args.working_directory + '/lmdb_converted'
pp_in = tools.pre_processing.PreProcessingInputCSV(args.file, delimiter = ',', 
                                                   label_column = 4,
                                                   label_column_mapping = {
                                                       'Iris-setosa': 0,
                                                       'Iris-versicolor': 1, 
                                                       'Iris-virginica': 2
                                                   })
pp_out_converted = tools.pre_processing.PreProcessingOutputLMDB(lmdb_converted)
pp_convert = tools.pre_processing.PreProcessingNormalize(pp_in, pp, 7.9)
pp_convert.run()    
    
print('LMDB:')
lmdb = tools.lmdb_io.LMDB(lmdb_converted)
images, labels, keys = lmdb.read()
    
for n in range(len(images)):
    print images[n].reshape((4)), labels[n]
```

- 참고 script : `examples/imagenet/create_imagenet.sh`

> [Training Multi-Layer Neural Network with Caffe](http://nbviewer.jupyter.org/github/joyofdata/joyofdata-articles/blob/master/deeplearning-with-caffe/Neural-Networks-with-Caffe-on-the-GPU.ipynb)



## 2. Data Augmentation

- tools.data_augmentation

## 3. Deploying Networks

training/testing .prototxt 파일로 네트워크를 Deply하려면 입력 LMDB 레이어 삭제 + Loo 레이어 삭제 필요 
- This transformation can be automated by `tools.prototxt.train2deploy`


## 4. Visualization 

## 4.1 네트워크 모델 그래프로 그리기 

- draw_net.py 이용 

```python 
python ./tools/caffe_tool_draw_net.py ./models/conv.prototxt ./models/my_net.png
from IPython.display import Image 
Image(filename='./models/my_net.png')
```

### 4.2 학습 결과 살펴보기 (제공 툴)

1. 학습 및 테스트 종료시 로그 파일은 /tmp 에 저장됨 (재 부팅시 삭제 되므로 백업 필요)
2. tools/extra/plot_training_log.py.example파일 이용하여 결과 그래프로 그리기 
    - plot_training_log.py.example {0=test, 6=train} {저장이미지 파일명} {입력 로그 파일명}   
    - eg. plot_training_log.py.example 6 train.png train.log
    
> 가끔 이미지 그래프가 직선이 그어지는 오류가 있는데 해당 log파일의 data layer prefetch queue empyt라고 적힌 부분 삭제 

### 4.3 학습 결과 살펴보기 (plot)

![](http://i.imgur.com/JDxFeT9.png)

```python
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Custom Test Accuracy: {:.2f}'.format(test_acc[-1]))

```

> 전체 코드 : [https://github.com/BVLC/caffe/blob/master/examples/01-learning-lenet.ipynb](https://github.com/BVLC/caffe/blob/master/examples/01-learning-lenet.ipynb)의 마지막 부분 
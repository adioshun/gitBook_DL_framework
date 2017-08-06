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

## 2. Data Augmentation

- tools.data_augmentation

## 3. Deploying Networks

training/testing .prototxt 파일로 네트워크를 Deply하려면 입력 LMDB 레이어 삭제 + Loo 레이어 삭제 필요 
- This transformation can be automated by `tools.prototxt.train2deploy`



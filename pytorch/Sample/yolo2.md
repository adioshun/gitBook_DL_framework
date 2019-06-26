|제목|YOLO2|
|-|-|
|코드|[marvis](https://github.com/marvis/pytorch-yolo2)|
|참고||

# PyTorch YOLO2

## 1. 개요 

## 2. 설치 

```bash
# pytorch 설치 
conda create -n py2torch python=2.7 ipykernel
source activate py2torch
conda install pytorch torchvision cuda80 -c soumith

#YOLO설치
git clone git@github.com:marvis/pytorch-yolo2.git
wget http://pjreddie.com/media/files/yolo.weights
python detect.py cfg/yolo.cfg yolo.weights data/dog.jpg
```

## 3. Training

#### Training YOLO on VOC
##### Get The Pascal VOC Data
```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```
##### Generate Labels for VOC
```
wget http://pjreddie.com/media/files/voc_label.py
python voc_label.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > voc_train.txt
```
##### Modify Cfg for Pascal Data
Change the cfg/voc.data config file
```
train  = train.txt
valid  = 2007_test.txt
names = data/voc.names
backup = backup
```
##### Download Pretrained Convolutional Weights
Download weights from the convolutional layers
```
wget http://pjreddie.com/media/files/darknet19_448.conv.23
```
or run the following command:
```
python partial.py cfg/darknet19_448.cfg darknet19_448.weights darknet19_448.conv.23 23
```
##### Train The Model
```
python train.py cfg/voc.data cfg/yolo-voc.cfg darknet19_448.conv.23
```
##### Evaluate The Model
```
python valid.py cfg/voc.data cfg/yolo-voc.cfg yolo-voc.weights
python scripts/voc_eval.py results/comp4_det_test_
```
mAP test on released models
```
yolo-voc.weights 544 0.7682 (paper: 78.6)
yolo-voc.weights 416 0.7513 (paper: 76.8)
tiny-yolo-voc.weights 416 0.5410 (paper: 57.1)

```
## 4. Testing 


## 5. Fine tuning  





--- 
변경 사항 
```python 
# detect.py

PATH_TO_TEST_IMAGES_DIR = '/home/hjlim99/test_images/'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'frame{}.jpg'.format(i)) for i in range(1, 11561) ]


def detect(cfgfile, weightfile, imgfile):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    for image_path in TEST_IMAGE_PATHS:
        img = Image.open(image_path).convert('RGB')
        sized = img.resize((m.width, m.height))
    
    
    #img = Image.open(imgfile).convert('RGB')
    #sized = img.resize((m.width, m.height))
    
        for i in range(2):
            start = time.time()
            boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
            finish = time.time()
            if i == 1:
                print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

        class_names = load_class_names(namesfile)
        #plot_boxes(img, boxes, 'predictions.jpg', class_names)
        filename = image_path.replace('/home/hjlim99/test_images/', '')
        plot_boxes(img, boxes, './result/{}'.format(filename), class_names)
        #'./save/{}.png'.format(image_path)


cfgfile = "cfg/yolo.cfg"
weightfile = "weight/yolo.weights"
imgfile = "data/dog.jpg"

detect(cfgfile, weightfile, imgfile)
```

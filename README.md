**Required Environment**

- Python 3.7
- PyTorch == 1.7.1

**Installation Steps**

```bash
git clone https://github.com/1377Zing/Yolov8_Sperm_detection.git
cd Yolov8_Sperm_detection
pip install -r requirements.txt
```

**Download Pretrained Weights**

You can download the pretrained weights for training from：
 https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s.pth

**Dataset Preparation**

Dataset Source: You can obtain the EVISAN dataset from：https://zenodo.org/records/4303768

Dataset Format: This project uses the VOC format for training. You need to prepare your own dataset before training.

Label Files: Place the .xml label files in the Annotation folder under VOCdevkit/VOC2007.

Image Files: Place the .jpg image files in the JPEGImages folder under VOCdevkit/VOC2007.


**Dataset Processing**

After arranging the dataset, you need to generate the 2007_train.txt and 2007_val.txt files for training using the voc_annotation.py script.

Modify the parameters in voc_annotation.py. Specifically, modify the classes_path parameter, which points to the .txt file corresponding to the detection classes.

When training your own dataset, you can create a cls_classes.txt file and write the classes you want to distinguish in it. For the EVISAN dataset, the content of the model_data/cls_classes.txt file should be:


```bash
black
```

**Network Training**

There are many training parameters in the train.py file. You can adjust them according to the comments. 

The most important part is that the classes_path in train.py should be the same as that used in voc_annotation.py. 

If you use the pretrained weights, modify the model_path parameter to the location of the weight file.

Run the following command to start training:

```bash
python train.py
```


# ArcFace_pytorch: A simple implementation of ArcFace model
**Reference:** https://github.com/deepinsight/insightface (Official repository of ArcFace) 

## Preparation
Create your dataset like the example below:<br/>
```
data
--YourDatasetName
  --Label 1
  ----image1.jpg
  ----image2.jpg
  ----...
  --Label 2
  ----...
```
You can find pretrained model at [link](https://drive.google.com/drive/folders/1FMXmo0I9Mhqgjn2cwyD9WcoaV0Ey65dU?usp=sharing) 

## Configuration 
```loss```: Now you can choose ```ArcFace``` or ```ElasticArcFace```. Several other losses are in progress. <br/>
```backbone```: ```irse50``` and ```mobilenet``` have pretrained model at [link](https://drive.google.com/drive/folders/1FMXmo0I9Mhqgjn2cwyD9WcoaV0Ey65dU?usp=sharing). Some other backbones are listed in the doctrings of ```ArcFaceModel``` class, but you have to train them from scratch. <br/>
```root_dir```: The path to the directory of dataset
```use_improved_optim```: use SAM Optimizer for training (set its value is ```true``` if you don't want to use Adam optimizer). You can find the original implementation at [link](https://github.com/davda54/sam) <br/>
To make advanced configurations, pay your attention to the docstring :)

## Training
run file ```main.py``` 


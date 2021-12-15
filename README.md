# Improved ArcFace: Some improvements on ArcFace model
**Reference:** https://github.com/deepinsight/insightface (Official repository of ArcFace) 

## Preparation
Create your dataset like the [sample dataset](https://drive.google.com/file/d/1D9Wt6horQdrFdRAMxc3CdYkKHulV00Up/view?usp=sharing):<br/>
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

You can find pretrained model at [link](https://drive.google.com/drive/folders/1FMXmo0I9Mhqgjn2cwyD9WcoaV0Ey65dU?usp=sharing) <br />

For face alignment, run ```$python align_face.py --root_dir /path/to/dataset/folder --dst_w 112 --dst_h 112``` (You can change the destination size ```--dst_w``` is output width and ```--dst_h``` is output height) 

## Configuration 
### Train
* ```loss```: Now you can choose ```ArcFace``` or ```ElasticArcFace```. Several other losses are in progress. <br/>
* ```backbone```: ```irse50``` and ```mobilenet``` have pretrained models on insightface's datasets at [link](https://drive.google.com/drive/folders/1FMXmo0I9Mhqgjn2cwyD9WcoaV0Ey65dU?usp=sharing). Some other backbones are listed in the doctring of ```ArcFaceModel``` class, but you have to train them from scratch. <br/>
* ```root_dir```: The path to the directory of train dataset <br/>
* ```use_sam_optim```: use SAM Optimizer for training (set its value is ```true``` if you don't want to use Adam optimizer). You can find the original implementation at [link](https://github.com/davda54/sam) <br/>
* ```use_lamb_optim```: use Lamb Optimizer for training (set its value is ```true``` if you don't want to use Adam optimizer). You can find the original implementation at [link](https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py) <br/>
* ```verbose```: 0: nothing will be shown; 1: shows results per epoch only; 2: shows train losses per iteration <br/>
<br/>
**Note**: Don't set the values of ```use_sam_optim``` and  ```use_sam_optim``` are ```true``` simultaneously. To make advanced configurations, pay your attention to the docstrings :)
### Test
* ```trainset_path```: It is the same as ```root_dir``` in training phase <br/>
* ```testset_path```: The path to the directory of test dataset <br/>

## Training
In terminal, run ```$ python main.py --config ./path/to/config/file.json --phase train --device 0```

## Testing
In terminal, run ```$ python main.py --config ./path/to/config/file.json --phase test --device 0```


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

For face alignment, run 
```bat
python align_face.py --root_dir /path/to/dataset/folder --dst_w 112 --dst_h 112
``` 
(You can change the destination size ```--dst_w``` is output width and ```--dst_h``` is output height) 

## Configuration 
### Train
* ```loss```: Now you can choose ```ArcFace``` or ```ElasticArcFace```. <br/>
* ```backbone```: Find supported backbone in ArcFaceModel's docstring. ```irse50``` and ```mobilefacenet``` have pretrained models on insightface's datasets at [link](https://drive.google.com/drive/folders/1FMXmo0I9Mhqgjn2cwyD9WcoaV0Ey65dU?usp=sharing). Other ones are listed in the doctring of ```ArcFaceModel``` class, but they have to be trained from scratch. <br/>
* ```freeze_model```: Freeze the backbone of trained model for transfer learning. Set its value is ```false``` to train from scratch.
* ```root_dir```: The path to the directory of train dataset <br/>
* ```use_lr_scheduler```: Use learning rate scheduler.
* ```optimizer```: Optimizer for training progress. ```sam```, ```lamb```, ```adam``` and ```adan``` are supported optimizer. You can find the original implementation of [SAM](https://github.com/davda54/sam), [LAMB](https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py) and [Adan](https://github.com/lucidrains/Adan-pytorch). If you don't change the config, the default optimizer is ADAM. <br/>
* ```verbose```: 0: nothing will be shown; 1: shows results per epoch only; 2: shows train losses per iteration <br/>
* ```prefix```: Prefix of saved model's name.

### Test
* ```trainset_path```: Path to the directory of dataset used for training the test model. <br/>
* ```testset_path```: Path to the directory of test dataset. <br/>
* ```pretrained_model_path```: Path to the pretrained model.

## Training
In terminal, run 
```bat 
python main.py --config ./path/to/config/file.json --phase train --device [gpu_id]
```
Currently, only training on a single gpu is supported.

## Testing
In terminal, run 
```bat
python main.py --config ./path/to/config/file.json --phase test --device [gpu_id]
```
## Verification
In terminal, run 
```bat
python verification.py --config ./path/to/config/file.json
```
Inference on single image is also supported, check the commented lines in the ```verification.py``` file. Filling hair is a miscellaneous option to observe the decrease of model's performance when the given face is partly covered.  

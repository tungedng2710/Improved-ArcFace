# ArcFace_pytorch: A simple implementation of ArcFace model
**Reference:** https://github.com/deepinsight/insightface

## Preparation
Create your dataset like the example below:<br/>
```
YourDatasetName
--Label 1
----image1.jpg
----image2.jpg
----...
--Label 2
----...
```
You can find pretrained model at [link](https://drive.google.com/drive/folders/1FMXmo0I9Mhqgjn2cwyD9WcoaV0Ey65dU?usp=sharing) 

## Training
In terminal, run ```$ python main.py --config configs/e_arcface.json --phase test```


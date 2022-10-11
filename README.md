# Faithfulness metrics for saliency maps 


TODO: explain the role of each script


This repository implements the faithfulness metrics mentionned in the paper "Computing and evaluating saliency maps for image classification: a tutorial" in Pytorch.
This can be used to compute the following metrics: 
- Deletion Area Under Curve (DAUC)/Insertion Area Under Curve (IAUC) [(Petsiuk et al. 2019)](https://arxiv.org/abs/1806.07421)
- Deletion Correlation (DC)/Insertion Correlation (IC) [(Gomez et al. 2022)](https://link.springer.com/chapter/10.1007/978-3-031-09037-0_8)
- Increase In Confidence (IIC)/Average Drop (AD) [(Chattopadhyay et al. 2017)](https://arxiv.org/abs/1710.11063)
- Avereage Drop in Deletion (ADD) [(Jung et al.)](https://arxiv.org/pdf/2102.05228.pdf)

![alt text](pics/metrics_repo_illust.png)

## Single step metrics

This section covers the use of the IIC, AD and ADD metrics.
First, generate the saliency map of the image:
```
saliency_map = gradcam.attribute(img,class_ind)
```
Then, compute the metric:

```
iic = IIC()
iic_mean = iic(model,data,explanations,class_to_explain)
```

The ```__call__()``` method for all the metrics requires the following arguments :
- ```model```: a ```torch.nn.Module``` that outputs a score tensor of shape (NxC), on which a softmax activation has been applied.
- ```data```: the input image tensor of shape (Nx3xHxW).
- ```explanations```: the saliency maps tensor of shape (Nx1xH'xW')
- ```class_to_explain```: The index of the class to explain for each input image. The shape shoud be (N).

The value returned by this method is simply the average value of the metric over all the images.

## Multi-step metrics 

This section covers the use of the DAUC, IAUC, DC and IC metrics.
These metrics work similarly but some argument have to be passed to the constructor:

```
dauc = DAUC(data_shape,explanation_shape,bound_max_step=True)
```
Where ```data_shape``` and ```explanation_shape``` are the shape of the image and saliency map tensor.
A high resolution saliency map of size 56x56 would require approximately 3k inferences.
To prevent too much computation, you can set the ```bound_max_step``` argument to True to limits the amount of computation that can be computed.
More precisly, this argument forces to mask/reveal several pixels before computing a new inference if the resolution of the saliency map is superior to 14x14.
The metric is then computed the same way as the single step metrics:

```
dauc_mean = dauc(model,data,explanations,class_to_explain)
```

## Demonstration

Look at the ```demo.ipynb``` script for a demonstration.
If you want to re-run the demo, you should download the [model's weights](https://drive.google.com/file/d/1JdHJjvCb9IAtcwKizo_KGLDR73UmtFYY/view?usp=sharing) pretrained on the CUB dataset and put it on the project's root.
You should also download the [CUB test dataset](http://www.vision.caltech.edu/datasets/cub_200_2011/) and put it in a "data" folder located at the project's root.
The dataset should be formated as expected by the ```torchvision.datasets.ImageFolder``` class from ```torchvision```.
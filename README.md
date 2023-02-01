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

This section covers the use of the Increase In Confidence (IIC), Average Drop (AD) and Average Drop in Deletion (ADD) metrics.
First, generate the saliency map of the image however you want. The only constraint is that the map should be a tensor of size (Nx1xH'xW'):
```
saliency_map = gradcam.attribute(img,class_ind)
```
Then, compute the metric. In this example, we use the class IIC_AD of this library to compute both the AD and IIC metric, as they require similar computations:

```
iic_ad = IIC_AD()
metric_dict = iic_ad(model,data,explanations,class_to_explain)
mean_iic = metric_dict["iic"]
mean_ad = metric_dict["ad"]
```

The resulting dictionary has two entries ```ic``` and ```ad``` which correspond to the mean value of the two metrics.

The ```__call__()``` method for all the metrics requires the following arguments :
- ```model```: a ```torch.nn.Module``` that outputs a score tensor of shape (NxC), on which a softmax activation has been applied.
- ```data```: the input image tensor of shape (Nx3xHxW).
- ```explanations```: the saliency maps tensor of shape (Nx1xH'xW')
- ```class_to_explain```: The index of the class to explain for each input image. The shape shoud be (N).

The ADD class is used similarly:
```
add = ADD()
metric_dict = add(model,data,explanations,class_to_explain)
mean = metric_dict["add"]
```
This resulting dictionary has only one entry ```add``` which correspond to the mean value of the ADD metric.

## Multi-step metrics 

This section covers the use of the DAUC, IAUC, DC and IC metrics.
These metrics work similarly but some argument have to be passed to the constructor:

```
deletion = Deletion(data_shape,explanation_shape,bound_max_step=True)
```
Where ```data_shape``` and ```explanation_shape``` are the shape of the image and saliency map tensor.
A high resolution saliency map of size 56x56 would require approximately 3k inferences.
To prevent too much computation, you can set the ```bound_max_step``` argument to True to limits the amount of computation that can be computed.
More precisly, this argument forces to mask/reveal several pixels before computing a new inference if the resolution of the saliency map is superior to 14x14.
The call method returns a dictionary with two entries: ```dauc``` (which correspond to the original Deletion metric by Petsiuk et al.) and ```dc``` (the variant proposed by Gomez et al.).

```
result_dic = deletion(model,data,explanations,class_to_explain)
dauc = result_dic["dauc"]
dc = result_dic["dc"]
```
The Insertion metric is computed similarly:

```
insertion = Insertion(data_shape,explanation_shape,bound_max_step=True)
result_dic = insertion(model,data,explanations,class_to_explain)
iauc = result_dic["iauc"]
ic = result_dic["ic"]
```

## Changing the filling method

These metrics work by removing parts of the image and replacing it with something else, for e.g. black pixels (Deletion, AD, IIC, ADD) or a blurred version of the input image (Insertion).
The default replacing method can be changed with the ```data_replace_method``` argument passed to the constructor:

```
insertion = Insertion(data_shape,explanation_shape,bound_max_step=True,data_replace_method="black")
add = ADD(data_replace_method="blur")
```
Currently, the autorized values are:
- "black": replaces with black pixels
- "blur": replaces with a blurred version of the input image

## Demonstration

Look at the ```demo.ipynb``` script for a demonstration.
If you want to re-run the demo, you should download the [model's weights](https://drive.google.com/file/d/1JdHJjvCb9IAtcwKizo_KGLDR73UmtFYY/view?usp=sharing) pretrained on the CUB dataset and put it on the project's root.
You should also download the [CUB test dataset](http://www.vision.caltech.edu/datasets/cub_200_2011/) and put it in a "data" folder located at the project's root.
The dataset should be formated as expected by the ```torchvision.datasets.ImageFolder``` class from ```torchvision```.